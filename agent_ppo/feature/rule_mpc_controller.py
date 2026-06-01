#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Strict linear MPC controller for waypoint navigation with yaw locked to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agent_ppo.conf.conf import Config
from agent_ppo.feature.observation_parser import ObservationParser
from agent_ppo.feature.obstacle_strategies import (
    ObstacleStrategyParams,
    STRATEGY_DEFAULT,
    get_strategy,
)

try:
    import osqp
except ImportError:  # pragma: no cover
    osqp = None

try:
    from scipy import sparse
except ImportError:  # pragma: no cover
    sparse = None


@dataclass
class AxisMPCResult:
    command: torch.Tensor
    predicted_state: np.ndarray


class LinearAxisMPC:
    """Single-axis linear MPC for relative-position state x=[r, v]^T.

    Relative-state dynamics with static target and yaw locked:
        r_{k+1} = r_k - dt * v_k - 0.5 * dt^2 * a_k
        v_{k+1} = v_k + dt * a_k

    Thus:
        x_{k+1} = A x_k + B u_k
    where
        x = [r, v]^T, u = a
        A in R^(2x2), B in R^(2x1)

    Prediction over horizon N:
        Y = Psi x0 + Theta U
    where
        Y in R^(2N), U in R^N

    Cost:
        J = (Y - Y_ref)^T Qbar (Y - Y_ref) + U^T Rbar U + dU^T Rdbar dU
    """

    def __init__(
        self,
        dt: float,
        horizon: int,
        q_pos: float,
        q_vel: float,
        r_u: float,
        r_du: float,
        a_max: float,
    ):
        if osqp is None or sparse is None:
            raise ImportError("Strict MPC requires both osqp and scipy to be installed.")

        self.dt = float(dt)
        self.horizon = int(horizon)
        self.a_max = float(a_max)
        self.r_du = float(r_du)

        self.A = np.array(
            [
                [1.0, -self.dt],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.B = np.array(
            [
                [-0.5 * self.dt * self.dt],
                [self.dt],
            ],
            dtype=np.float64,
        )

        self.Psi, self.Theta = self._build_prediction_matrices()

        q_stage = np.diag([q_pos, q_vel]).astype(np.float64)
        q_bar = sparse.block_diag([q_stage] * self.horizon, format="csc")
        r_bar = sparse.eye(self.horizon, format="csc") * float(r_u)

        d_mat = np.eye(self.horizon, dtype=np.float64)
        for i in range(1, self.horizon):
            d_mat[i, i - 1] = -1.0
        self.d_mat = d_mat
        rd_bar = sparse.csc_matrix(d_mat.T @ d_mat * float(r_du))

        theta_sparse = sparse.csc_matrix(self.Theta)
        hessian = theta_sparse.T @ q_bar @ theta_sparse + r_bar + rd_bar
        hessian = (hessian + hessian.T) * 0.5

        # Box constraint: -a_max <= U <= a_max.
        a_cons = sparse.eye(self.horizon, format="csc")
        lower = np.full(self.horizon, -self.a_max, dtype=np.float64)
        upper = np.full(self.horizon, self.a_max, dtype=np.float64)

        self._q_bar = q_bar
        self._theta_sparse = theta_sparse
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=hessian,
            q=np.zeros(self.horizon, dtype=np.float64),
            A=a_cons,
            l=lower,
            u=upper,
            verbose=False,
            warm_start=True,
            polish=True,
            eps_abs=1.0e-4,
            eps_rel=1.0e-4,
            max_iter=4000,
        )

    def _build_prediction_matrices(self):
        psi = np.zeros((2 * self.horizon, 2), dtype=np.float64)
        theta = np.zeros((2 * self.horizon, self.horizon), dtype=np.float64)
        for i in range(self.horizon):
            a_power = np.linalg.matrix_power(self.A, i + 1)
            psi[2 * i : 2 * i + 2, :] = a_power
            for j in range(i + 1):
                a_term = np.linalg.matrix_power(self.A, i - j)
                theta[2 * i : 2 * i + 2, j : j + 1] = a_term @ self.B
        return psi, theta

    def solve(self, x0: np.ndarray, u_prev: float, ref_pos: float = 0.0, ref_vel: float = 0.0) -> AxisMPCResult:
        y_ref = np.tile(np.array([ref_pos, ref_vel], dtype=np.float64), self.horizon)
        y_free = self.Psi @ x0

        # dU = D U - [u_{k-1}, 0, ..., 0]^T
        du_offset = np.zeros(self.horizon, dtype=np.float64)
        du_offset[0] = float(u_prev)

        linear_term = self._theta_sparse.T @ (self._q_bar @ (y_free - y_ref))
        linear_term = np.asarray(linear_term).reshape(-1)
        linear_term -= (self.d_mat.T @ du_offset) * self.r_du

        self._solver.update(q=linear_term)
        result = self._solver.solve()
        if result.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP failed with status: {result.info.status}")

        u_seq = result.x
        x_pred = y_free + self.Theta @ u_seq
        return AxisMPCResult(command=torch.tensor(u_seq[0], dtype=torch.float32), predicted_state=x_pred[:2])


class RuleMPCController:
    """Strict linear MPC waypoint controller with yaw disabled."""

    def __init__(self, logger=None):
        self.logger = logger
        self.parser = ObservationParser()
        self._mpc_xy = LinearAxisMPC(
            dt=Config.MPC_DT,
            horizon=Config.MPC_HORIZON,
            q_pos=Config.MPC_Q_POS_XY,
            q_vel=Config.MPC_Q_VEL_XY,
            r_u=Config.MPC_R_U_XY,
            r_du=Config.MPC_R_DU,
            a_max=Config.MPC_A_MAX_XY,
        )
        self._mpc_z = LinearAxisMPC(
            dt=Config.MPC_DT,
            horizon=Config.MPC_HORIZON,
            q_pos=Config.MPC_Q_POS_Z,
            q_vel=Config.MPC_Q_VEL_Z,
            r_u=Config.MPC_R_U_Z,
            r_du=Config.MPC_R_DU_Z,
            a_max=Config.MPC_A_MAX_Z,
        )
        self._mpc_xy_hover = LinearAxisMPC(
            dt=Config.MPC_DT,
            horizon=Config.MPC_HOVER_HORIZON,
            q_pos=Config.MPC_HOVER_Q_POS_XY,
            q_vel=Config.MPC_HOVER_Q_VEL_XY,
            r_u=Config.MPC_HOVER_R_U_XY,
            r_du=Config.MPC_HOVER_R_DU,
            a_max=Config.MPC_A_MAX_XY,
        )
        self._mpc_z_hover = LinearAxisMPC(
            dt=Config.MPC_DT,
            horizon=Config.MPC_HOVER_HORIZON,
            q_pos=Config.MPC_HOVER_Q_POS_Z,
            q_vel=Config.MPC_HOVER_Q_VEL_Z,
            r_u=Config.MPC_HOVER_R_U_Z,
            r_du=Config.MPC_HOVER_R_DU_Z,
            a_max=Config.MPC_A_MAX_Z,
        )

        self._prev_u = None
        self._last_target_slot = None
        self._startup_counter = None
        self._yaw_align_hold_counter = None
        self._locked_target_slot = None
        self._wp_hold_counter = None
        self._avoid_obstacle_slot = None
        self._avoid_side = None
        self._avoid_hold_counter = None
        self._segment_length = None
        self._tide_hold_counter = None
        self._avoid_blend_weight = None
        self._avoid_hold_total = None
        self._tan_direction_lock = None
        self._tan_lock_counter = None
        self._hard_safety_counter = None
        self._post_danger_guard_counter = None
        self._obs_count = 0
        self._strategy: ObstacleStrategyParams = STRATEGY_DEFAULT
        self._obs_count_logged = False
        self._log_counter = 0

    # ------------------------------------------------------------------
    # Obstacle-count strategy selection
    # ------------------------------------------------------------------

    def _update_obs_strategy(self, parsed):
        """Detect active obstacle count and select the matching strategy."""
        active = parsed.obstacle_active
        counts = active.sum(dim=-1).int()
        new_count = int(counts.float().median().round().item())
        new_count = max(new_count, 0)
        if new_count != self._obs_count:
            self._obs_count = new_count
            self._strategy = get_strategy(new_count)
            self._obs_count_logged = False

        if not self._obs_count_logged:
            s = self._strategy
            L = self.logger.info if self.logger else print
            L(
                f"[RULE-MPC STRATEGY] obs_count={self._obs_count} "
                f"speed_cap={s.wp_speed_cap:.2f} "
                f"danger_clr={s.danger_segment_clearance:.2f} "
                f"obs_margin={s.obs_base_margin:.2f} "
                f"brake_safe={s.brake_safe_speed:.2f} "
                f"hard_clr={s.hard_safety_clearance:.2f}"
            )
            self._obs_count_logged = True

    def _prepare_obs(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        original_shape = None
        if obs.dim() == 3:
            original_shape = obs.shape
            obs = obs.view(obs.shape[0] * obs.shape[1], -1)
        return obs, original_shape

    @staticmethod
    def _restore_shape(action: torch.Tensor, original_shape):
        if original_shape is None:
            return action
        return action.view(original_shape[0], original_shape[1], -1)

    @staticmethod
    def _extract_attitude(rotation_matrix: torch.Tensor):
        r20 = torch.clamp(rotation_matrix[:, 2, 0], -0.999, 0.999)
        pitch = torch.asin(-r20)
        roll = torch.atan2(rotation_matrix[:, 2, 1], rotation_matrix[:, 2, 2])
        yaw = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
        return roll, pitch, yaw

    def _ensure_state(self, batch_size: int, device: torch.device):
        if self._prev_u is None or self._prev_u.shape[0] != batch_size:
            self._prev_u = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
            self._last_target_slot = torch.full((batch_size,), -2, dtype=torch.long, device=device)
            self._startup_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._yaw_align_hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._locked_target_slot = torch.full((batch_size,), -2, dtype=torch.long, device=device)
            self._wp_hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._avoid_obstacle_slot = torch.full((batch_size,), -1, dtype=torch.long, device=device)
            self._avoid_side = torch.ones(batch_size, dtype=torch.float32, device=device)
            self._avoid_hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._segment_length = torch.zeros(batch_size, dtype=torch.float32, device=device)
            self._tide_hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._avoid_blend_weight = torch.zeros(batch_size, dtype=torch.float32, device=device)
            self._avoid_hold_total = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._tan_direction_lock = torch.zeros(batch_size, dtype=torch.float32, device=device)
            self._tan_lock_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._hard_safety_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._post_danger_guard_counter = torch.zeros(batch_size, dtype=torch.long, device=device)

    def reset_envs(self, env_mask: torch.Tensor):
        if self._prev_u is None or env_mask is None:
            return
        if env_mask.dtype != torch.bool:
            env_mask = env_mask.to(dtype=torch.bool)
        self._prev_u[env_mask] = 0.0
        self._last_target_slot[env_mask] = -2
        self._startup_counter[env_mask] = 0
        self._yaw_align_hold_counter[env_mask] = 0
        self._locked_target_slot[env_mask] = -2
        self._wp_hold_counter[env_mask] = 0
        self._avoid_obstacle_slot[env_mask] = -1
        self._avoid_side[env_mask] = 1.0
        self._avoid_hold_counter[env_mask] = 0
        self._segment_length[env_mask] = 0.0
        self._tide_hold_counter[env_mask] = 0
        self._avoid_blend_weight[env_mask] = 0.0
        self._avoid_hold_total[env_mask] = 0
        self._tan_direction_lock[env_mask] = 0.0
        self._tan_lock_counter[env_mask] = 0
        self._hard_safety_counter[env_mask] = 0
        self._post_danger_guard_counter[env_mask] = 0

    def reset(self, batch_size: int = None, device: torch.device = None):
        if batch_size is not None and device is not None:
            self._prev_u = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
            self._last_target_slot = torch.full((batch_size,), -2, dtype=torch.long, device=device)
            self._startup_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._yaw_align_hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._locked_target_slot = torch.full((batch_size,), -2, dtype=torch.long, device=device)
            self._wp_hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._avoid_obstacle_slot = torch.full((batch_size,), -1, dtype=torch.long, device=device)
            self._avoid_side = torch.ones(batch_size, dtype=torch.float32, device=device)
            self._avoid_hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._segment_length = torch.zeros(batch_size, dtype=torch.float32, device=device)
            self._tide_hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._avoid_blend_weight = torch.zeros(batch_size, dtype=torch.float32, device=device)
            self._avoid_hold_total = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._tan_direction_lock = torch.zeros(batch_size, dtype=torch.float32, device=device)
            self._tan_lock_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._hard_safety_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._post_danger_guard_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            self._prev_u = None
            self._last_target_slot = None
            self._startup_counter = None
            self._yaw_align_hold_counter = None
            self._locked_target_slot = None
            self._wp_hold_counter = None
            self._avoid_obstacle_slot = None
            self._avoid_side = None
            self._avoid_hold_counter = None
            self._segment_length = None
            self._tide_hold_counter = None
            self._avoid_blend_weight = None
            self._avoid_hold_total = None
            self._tan_direction_lock = None
            self._tan_lock_counter = None
            self._hard_safety_counter = None
            self._post_danger_guard_counter = None

    @staticmethod
    def _segment_clearance_to_targets(
        obstacle_xy: torch.Tensor,
        obstacle_radius: torch.Tensor,
        target_xy: torch.Tensor,
    ):
        if target_xy.numel() == 0:
            return torch.zeros(0, dtype=target_xy.dtype, device=target_xy.device)
        if obstacle_xy.numel() == 0:
            return torch.full((target_xy.shape[0],), float("inf"), dtype=target_xy.dtype, device=target_xy.device)

        target_norm_sq = torch.sum(target_xy * target_xy, dim=-1).clamp_min(1.0e-6)
        proj = obstacle_xy @ target_xy.transpose(0, 1)
        t = torch.clamp(proj / target_norm_sq.unsqueeze(0), 0.0, 1.0)
        closest = t.unsqueeze(-1) * target_xy.unsqueeze(0)
        dist = torch.norm(obstacle_xy.unsqueeze(1) - closest, dim=-1)
        clearance = dist - obstacle_radius.unsqueeze(-1)
        return torch.min(clearance, dim=0).values

    @staticmethod
    def _point_clearance_to_obstacles(
        obstacle_xy: torch.Tensor,
        obstacle_radius: torch.Tensor,
        point_xy: torch.Tensor,
    ):
        """Min radial clearance from a point to all obstacle surfaces (dist - radius)."""
        if point_xy.numel() == 0:
            return torch.zeros(0, dtype=point_xy.dtype, device=point_xy.device)
        if obstacle_xy.numel() == 0:
            return torch.full((point_xy.shape[0],), float("inf"), dtype=point_xy.dtype, device=point_xy.device)

        dist = torch.norm(obstacle_xy.unsqueeze(1) - point_xy.unsqueeze(0), dim=-1)
        clearance = dist - obstacle_radius.unsqueeze(-1)
        return torch.min(clearance, dim=0).values

    @staticmethod
    def _late_route_waypoint_mask(points: torch.Tensor, goal_rpos: torch.Tensor):
        """Filter waypoints that are on a reasonable route to the goal."""
        if points.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=points.device)

        goal_xy = goal_rpos[:2]
        points_xy = points[:, :2]
        goal_dist = torch.norm(goal_xy).clamp_min(1.0e-6)
        wp_dist = torch.norm(points_xy, dim=-1)
        wp_to_goal = torch.norm(points_xy - goal_xy.unsqueeze(0), dim=-1)
        route_extra = wp_dist + wp_to_goal - goal_dist
        projection = (points_xy @ goal_xy) / goal_dist

        max_extra = float(getattr(Config, "MPC_LATE_WP_ROUTE_EXTRA", 0.60))
        overshoot = float(getattr(Config, "MPC_LATE_WP_ROUTE_OVERSHOOT", 0.20))
        return (route_extra <= max_extra) & (projection >= 0.0) & (projection <= goal_dist + overshoot)

    def _build_dangerous_segment_mask(self, parsed, target_rpos: torch.Tensor, target_slot: torch.Tensor):
        danger_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        if not getattr(Config, "MPC_DANGER_SEGMENT_ENABLE", True):
            return danger_mask

        obs_count = self._obs_count
        if obs_count >= 4:
            clearance_limit = self._strategy.danger_segment_clearance
        elif obs_count == 3:
            clearance_limit = float(getattr(Config, "MPC_DANGER_SEGMENT_CLEARANCE", 0.35))
        else:
            clearance_limit = float(getattr(Config, "MPC_DANGER_SEGMENT_CLEARANCE", 0.35))

        for b in range(parsed.batch_size):
            slot = int(target_slot[b].item())
            if obs_count >= 3:
                if slot < -1:
                    continue
            else:
                if slot < 0:
                    continue
            active_obs = parsed.obstacle_active[b]
            if not active_obs.any():
                continue
            clearance = self._segment_clearance_to_targets(
                parsed.obstacle_rpos[b, active_obs, :2],
                parsed.obstacle_radius[b, active_obs],
                target_rpos[b : b + 1, :2],
            )
            danger_mask[b] = clearance[0] < clearance_limit
        return danger_mask

    def _select_local_target(self, parsed):
        batch_size = parsed.batch_size
        device = parsed.device

        target_rpos = parsed.goal_rpos.clone()
        target_slot = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        hover_mask = parsed.phase_hover > 0.5
        target_rpos[hover_mask] = parsed.goal_rpos[hover_mask]
        target_slot[hover_mask] = -99

        pending_mask = parsed.waypoint_active & (~parsed.waypoint_visited)
        if getattr(Config, "DIRECT_GOAL_MODE", False):
            return target_rpos, target_slot, hover_mask

        obs_count = self._obs_count

        # Dense / 456 path pre-reads
        if obs_count >= 3:
            frame_idx = int(self._log_counter)
            safe_frame = int(getattr(Config, "MPC_LATE_WP_SAFE_FRAME", 1050))
            force_goal_frame = int(getattr(Config, "MPC_FORCE_GOAL_FRAME", 1200))
            if obs_count >= 4:
                critical_clearance = self._strategy.critical_wp_clearance
                danger_wp_clearance_dense = self._strategy.danger_wp_clearance
            else:
                critical_clearance = float(getattr(Config, "MPC_CRITICAL_WP_CLEARANCE", 0.10))
                danger_wp_clearance_dense = float(getattr(Config, "MPC_DANGER_WP_CLEARANCE_DENSE", 0.35))
                skip_start_side_frame = int(getattr(Config, "MPC_SKIP_START_SIDE_FRAME", 950))
                skip_start_side_x = float(getattr(Config, "MPC_SKIP_START_SIDE_X", 2.0))

        for b in range(batch_size):
            if hover_mask[b]:
                self._locked_target_slot[b] = -99
                continue

            pending_idx = pending_mask[b].nonzero(as_tuple=False).view(-1)

            # ---- dense / 456 path (3+ obstacles): frame-based + clearance filtering ----
            if obs_count >= 3:
                if frame_idx > force_goal_frame:
                    self._locked_target_slot[b] = -1
                    continue
                if pending_idx.numel() == 0:
                    self._locked_target_slot[b] = -1
                    continue

                pending_points = parsed.waypoint_rpos[b, pending_idx]
                valid = torch.ones(pending_idx.shape[0], dtype=torch.bool, device=device)
                point_clearances = None
                segment_clearances = None

                # Dense-v3 only: skip start-side waypoints in early frames
                if obs_count == 3 and frame_idx > skip_start_side_frame:
                    drone_xy = self._drone_xy_from_start(parsed)[b]
                    waypoint_abs_x = drone_xy[0] + pending_points[:, 0]
                    valid = valid & (waypoint_abs_x >= skip_start_side_x)

                if parsed.obstacle_active[b].any():
                    active_obs = parsed.obstacle_active[b]
                    obstacle_xy = parsed.obstacle_rpos[b, active_obs, :2]
                    obstacle_radius = parsed.obstacle_radius[b, active_obs]
                    point_clearances = self._point_clearance_to_obstacles(
                        obstacle_xy, obstacle_radius, pending_points[:, :2],
                    )
                    valid = valid & (point_clearances >= critical_clearance)
                    segment_clearances = self._segment_clearance_to_targets(
                        obstacle_xy, obstacle_radius, pending_points[:, :2],
                    )

                if frame_idx > safe_frame:
                    route_mask = self._late_route_waypoint_mask(pending_points, parsed.goal_rpos[b])
                    safe_mask = valid.clone()
                    if parsed.obstacle_active[b].any():
                        segment_limit = float(getattr(Config, "MPC_DANGER_SEGMENT_CLEARANCE", 0.35))
                        safe_mask = safe_mask & (point_clearances >= danger_wp_clearance_dense) & (segment_clearances >= segment_limit)
                    valid = valid & route_mask & safe_mask

                if not valid.any():
                    self._locked_target_slot[b] = -1
                    continue

                pending_idx = pending_idx[valid]
                pending_points = pending_points[valid]

            # ---- sparse path (0/1/2 obstacles): simple nearest-neighbor ----
            else:
                if pending_idx.numel() == 0:
                    self._locked_target_slot[b] = -1
                    continue
                pending_points = parsed.waypoint_rpos[b, pending_idx]

            distances = torch.norm(pending_points, dim=-1)
            scores = distances.clone()

            if getattr(Config, "MPC_DANGER_SEGMENT_ENABLE", True) and parsed.obstacle_active[b].any():
                active_obs = parsed.obstacle_active[b]
                clearances = self._segment_clearance_to_targets(
                    parsed.obstacle_rpos[b, active_obs, :2],
                    parsed.obstacle_radius[b, active_obs],
                    pending_points[:, :2],
                )
                clearance_limit = float(getattr(Config, "MPC_DANGER_SEGMENT_CLEARANCE", 0.35))
                hard_clearance = float(getattr(Config, "MPC_DANGER_SEGMENT_HARD_CLEARANCE", 0.08))
                risk_weight = float(getattr(Config, "MPC_DANGER_SEGMENT_RISK_WEIGHT", 8.0))
                hard_penalty = float(getattr(Config, "MPC_DANGER_SEGMENT_HARD_PENALTY", 40.0))
                risk_gap = torch.clamp(clearance_limit - clearances, min=0.0)
                hard_gap = torch.clamp(hard_clearance - clearances, min=0.0)
                scores = scores + risk_weight * risk_gap * risk_gap + hard_penalty * hard_gap

            best_local = torch.argmin(scores)
            best_idx = int(pending_idx[best_local].item())
            target_rpos[b] = parsed.waypoint_rpos[b, best_idx]
            target_slot[b] = best_idx
            self._locked_target_slot[b] = best_idx

        return target_rpos, target_slot, hover_mask

    def _update_target_lock(self, parsed, target_rpos: torch.Tensor, target_slot: torch.Tensor, hover_mask: torch.Tensor):
        del parsed, target_rpos, hover_mask
        self._locked_target_slot.copy_(target_slot)

    def _build_dangerous_waypoint_mask(self, parsed, target_slot: torch.Tensor):
        danger_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        if not getattr(Config, "MPC_DANGER_WP_ENABLE", True):
            return danger_mask

        clearance_limit = float(getattr(Config, "MPC_DANGER_WP_CLEARANCE", 0.25))
        for b in range(parsed.batch_size):
            slot = int(target_slot[b].item())
            if slot < 0 or slot >= parsed.waypoint_rpos.shape[1]:
                continue
            if not parsed.waypoint_active[b, slot]:
                continue

            active_obs = parsed.obstacle_active[b]
            if not active_obs.any():
                continue

            wp_xy = parsed.waypoint_rpos[b, slot, :2]
            obs_xy = parsed.obstacle_rpos[b, active_obs, :2]
            obs_radius = parsed.obstacle_radius[b, active_obs]
            clearance = torch.norm(obs_xy - wp_xy.unsqueeze(0), dim=-1) - obs_radius
            danger_mask[b] = torch.min(clearance) < clearance_limit

        return danger_mask

    def _build_critical_waypoint_mask(self, parsed, target_slot: torch.Tensor):
        """Waypoints within MPC_CRITICAL_WP_CLEARANCE of an obstacle surface (dense-only)."""
        critical_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        clearance_limit = float(getattr(Config, "MPC_CRITICAL_WP_CLEARANCE", 0.05))
        for b in range(parsed.batch_size):
            slot = int(target_slot[b].item())
            if slot < 0 or slot >= parsed.waypoint_rpos.shape[1]:
                continue
            if not parsed.waypoint_active[b, slot]:
                continue

            active_obs = parsed.obstacle_active[b]
            if not active_obs.any():
                continue

            wp_xy = parsed.waypoint_rpos[b, slot, :2]
            obs_xy = parsed.obstacle_rpos[b, active_obs, :2]
            obs_radius = parsed.obstacle_radius[b, active_obs]
            clearance = torch.norm(obs_xy - wp_xy.unsqueeze(0), dim=-1) - obs_radius
            critical_mask[b] = torch.min(clearance) < clearance_limit

        return critical_mask

    def _hold_recent_dangerous_waypoint(
        self,
        parsed,
        target_rpos: torch.Tensor,
        target_slot: torch.Tensor,
        hover_mask: torch.Tensor,
        v_world: torch.Tensor,
    ):
        prev_slot = self._last_target_slot.clone()
        prev_danger = self._build_dangerous_waypoint_mask(parsed, prev_slot)

        # Dense paths (3+ obstacles): exclude critically-close WPs from hold (prevents deadlock)
        if self._obs_count >= 3:
            prev_critical = self._build_critical_waypoint_mask(parsed, prev_slot)
            changed_from_danger = (prev_slot >= 0) & prev_danger & (~prev_critical) & (target_slot != prev_slot) & (~hover_mask)
        else:
            changed_from_danger = (prev_slot >= 0) & prev_danger & (target_slot != prev_slot) & (~hover_mask)

        if not changed_from_danger.any():
            return target_rpos, target_slot, prev_danger, torch.zeros_like(changed_from_danger)

        stop_speed = float(getattr(Config, "MPC_DANGER_WP_STOP_SPEED", 0.08))
        hold_frames = max(int(getattr(Config, "MPC_DANGER_WP_HOLD_FRAMES", 8)), 0)
        speed_xy = torch.norm(v_world[:, :2], dim=-1)
        hold_mask = changed_from_danger & (
            (speed_xy > stop_speed) | (self._wp_hold_counter < hold_frames)
        )
        if not hold_mask.any():
            return target_rpos, target_slot, prev_danger, hold_mask

        target_held = target_rpos.clone()
        slot_held = target_slot.clone()
        for b in hold_mask.nonzero(as_tuple=False).view(-1):
            b_idx = int(b.item())
            slot = int(prev_slot[b_idx].item())
            target_held[b_idx] = parsed.waypoint_rpos[b_idx, slot]
            slot_held[b_idx] = slot
        self._wp_hold_counter[hold_mask] += 1
        return target_held, slot_held, prev_danger, hold_mask

    def _solve_axis(
        self,
        controller: LinearAxisMPC,
        position_error: float,
        velocity_world: float,
        prev_u: float,
        ref_vel: float = 0.0,
    ):
        x0 = np.array([position_error, velocity_world], dtype=np.float64)
        return controller.solve(x0=x0, u_prev=prev_u, ref_pos=0.0, ref_vel=ref_vel)

    def _build_reference_velocity(
        self,
        parsed,
        target_rpos: torch.Tensor,
        target_slot: torch.Tensor,
        hover_mask: torch.Tensor,
        yaw_align_mask: torch.Tensor,
        avoid_mask: torch.Tensor,
        near_goal_mask: torch.Tensor,
        dangerous_wp_mask: torch.Tensor,
        dangerous_segment_mask: torch.Tensor,
        post_danger_guard_mask: torch.Tensor = None,
    ):
        ref_vel = torch.zeros_like(target_rpos)
        danger_wp_brake_mask = torch.zeros(target_rpos.shape[0], dtype=torch.bool, device=target_rpos.device)
        if not getattr(Config, "MPC_WP_PASS_THROUGH_ENABLE", True):
            return ref_vel, danger_wp_brake_mask

        obs_count = self._obs_count

        wp_mask = (
            (target_slot >= 0)
            & (~hover_mask)
            & (~yaw_align_mask)
            & (~avoid_mask)
            & (~near_goal_mask)
        )

        # Dense / 456 path: goal-directed nav speed control
        if obs_count >= 3 and getattr(Config, "MPC_GOAL_SPEED_ENABLE", True):
            goal_nav_mask = (
                (target_slot == -1)
                & (~hover_mask)
                & (~yaw_align_mask)
                & (~avoid_mask)
                & (~dangerous_segment_mask)
            )
        else:
            goal_nav_mask = torch.zeros_like(hover_mask, dtype=torch.bool)

        if not wp_mask.any() and not goal_nav_mask.any():
            return ref_vel, danger_wp_brake_mask

        dist_xy = torch.norm(target_rpos[:, :2], dim=-1)
        dist = torch.norm(target_rpos, dim=-1)
        if wp_mask.any():
            changed_target = target_slot != self._last_target_slot
            refresh_segment = wp_mask & (changed_target | (self._segment_length <= 1.0e-4) | (dist > self._segment_length))
            self._segment_length[refresh_segment] = dist[refresh_segment].clamp_min(0.20)

        segment_len = torch.maximum(self._segment_length, dist).clamp_min(0.20)
        progress = torch.clamp(1.0 - dist / segment_len, 0.0, 1.0)

        # ---- Speed profile params: strategy (456) or Config (sparse/dense-v3) ----
        if obs_count >= 4:
            accel_frac = self._strategy.wp_accel_frac
            decel_frac = self._strategy.wp_decel_frac
            finish_frac = self._strategy.wp_finish_speed_frac
            speed_base = self._strategy.wp_speed_base
            speed_gain = self._strategy.wp_speed_gain
            speed_cap = self._strategy.wp_speed_cap
            min_speed = self._strategy.wp_min_speed
        else:
            accel_frac = max(float(getattr(Config, "MPC_WP_PROFILE_ACCEL_FRAC", 0.25)), 1.0e-3)
            decel_frac = max(float(getattr(Config, "MPC_WP_PROFILE_DECEL_FRAC", 0.25)), 1.0e-3)
            finish_frac = float(getattr(Config, "MPC_WP_PROFILE_FINISH_SPEED_FRAC", 0.32))
            speed_base = float(getattr(Config, "MPC_WP_PROFILE_MAX_SPEED_BASE", 0.45))
            speed_gain = float(getattr(Config, "MPC_WP_PROFILE_MAX_SPEED_GAIN", 0.42))
            speed_cap = float(getattr(Config, "MPC_WP_PROFILE_MAX_SPEED_CAP", 1.35))
            min_speed = float(getattr(Config, "MPC_WP_PROFILE_MIN_SPEED", 0.28))

        accel_scale = torch.clamp(progress / accel_frac, 0.0, 1.0)
        decel_dist = (segment_len * decel_frac).clamp_min(1.0e-3)
        decel_scale = finish_frac + (1.0 - finish_frac) * torch.clamp(dist / decel_dist, 0.0, 1.0)

        # Sparse path only: final waypoint carries momentum toward goal
        if obs_count <= 2:
            final_finish_frac = float(getattr(Config, "MPC_WP_PROFILE_FINAL_FINISH_SPEED_FRAC", 0.60))
            if final_finish_frac != finish_frac:
                pending = parsed.waypoint_active & (~parsed.waypoint_visited)
                pending_count = pending.sum(dim=1)
                slot_valid = target_slot >= 0
                slot_clamped = target_slot.clamp(min=0)
                slot_is_pending = torch.gather(pending, 1, slot_clamped.unsqueeze(1)).squeeze(1) & slot_valid
                final_approach = (pending_count == 1) & wp_mask & slot_is_pending
                if final_approach.any():
                    final_val = final_finish_frac + (1.0 - final_finish_frac) * torch.clamp(
                        dist[final_approach] / decel_dist[final_approach], 0.0, 1.0
                    )
                    decel_scale[final_approach] = final_val

        speed_scale = torch.minimum(accel_scale, decel_scale)
        max_speed = torch.clamp(speed_base + speed_gain * segment_len, min=min_speed, max=speed_cap)

        direction = target_rpos / dist.unsqueeze(-1).clamp_min(1.0e-6)
        if getattr(Config, "MPC_WP_PREVIEW_ENABLE", True):
            if obs_count >= 3:
                preview_mask = wp_mask & (~dangerous_wp_mask) & (~dangerous_segment_mask)
            else:
                preview_mask = wp_mask & (~dangerous_segment_mask)
            direction = self._blend_next_waypoint_direction(parsed, direction, target_rpos, target_slot, dist, preview_mask)

        # ---- Dense / 456 speed scaling layers ----
        if obs_count >= 3:
            if obs_count >= 4:
                danger_speed_scale = self._strategy.post_danger_speed_scale  # reused for danger segment scaling
                safe_speed_scale = self._strategy.safe_segment_speed_scale
                post_danger_scale = self._strategy.post_danger_speed_scale
                danger_wp_approach_scale = self._strategy.danger_wp_approach_speed_scale
                brake_frac = self._strategy.danger_wp_brake_dist_frac
            else:
                danger_speed_scale = float(getattr(Config, "MPC_DANGER_SEGMENT_SPEED_SCALE", 0.55))
                safe_speed_scale = float(getattr(Config, "MPC_SAFE_SEGMENT_SPEED_SCALE", 1.40))
                post_danger_scale = float(getattr(Config, "MPC_POST_DANGER_SEGMENT_SPEED_SCALE", 0.45))
                danger_wp_approach_scale = float(getattr(Config, "MPC_DANGER_WP_APPROACH_SPEED_SCALE", 0.75))
                brake_frac = float(getattr(Config, "MPC_DANGER_WP_BRAKE_DIST_FRAC", 0.33))

            danger_speed_scale_t = torch.where(
                dangerous_segment_mask,
                torch.full_like(max_speed, danger_speed_scale),
                torch.ones_like(max_speed),
            )
            safe_speed_scale_t = torch.where(
                (~dangerous_wp_mask) & (~dangerous_segment_mask) & (~post_danger_guard_mask),
                torch.full_like(max_speed, safe_speed_scale),
                torch.ones_like(max_speed),
            )
            post_danger_speed_scale_t = torch.where(
                post_danger_guard_mask,
                torch.full_like(max_speed, post_danger_scale),
                torch.ones_like(max_speed),
            )
            danger_wp_speed_scale_t = torch.where(
                dangerous_wp_mask,
                torch.full_like(max_speed, danger_wp_approach_scale),
                torch.ones_like(max_speed),
            )
            danger_wp_brake_mask = dangerous_wp_mask & (dist <= segment_len * brake_frac)

        active = wp_mask & (dist > 0.05)
        if obs_count >= 3:
            approach_mask = active & (~danger_wp_brake_mask)
            ref_speed = (
                max_speed * speed_scale
                * danger_speed_scale_t
                * safe_speed_scale_t
                * danger_wp_speed_scale_t
                * post_danger_speed_scale_t
            )
            ref_vel[approach_mask] = direction[approach_mask] * ref_speed[approach_mask].unsqueeze(-1)
        else:
            ref_vel[active] = direction[active] * (max_speed[active] * speed_scale[active]).unsqueeze(-1)

        # Dense / 456 path: goal-directed speed management
        if goal_nav_mask.any():
            if obs_count >= 4:
                fast_dist = float(getattr(Config, "MPC_GOAL_FAST_DIST", 0.60))
                brake_dist = float(getattr(Config, "MPC_GOAL_BRAKE_DIST", 0.45))
                precise_dist = float(getattr(Config, "MPC_GOAL_PRECISE_DIST", 0.30))
                fast_scale = self._strategy.goal_fast_speed_scale
                brake_scale = self._strategy.goal_brake_speed_scale
            else:
                fast_dist = float(getattr(Config, "MPC_GOAL_FAST_DIST", 0.60))
                brake_dist = float(getattr(Config, "MPC_GOAL_BRAKE_DIST", 0.45))
                precise_dist = float(getattr(Config, "MPC_GOAL_PRECISE_DIST", 0.30))
                fast_scale = float(getattr(Config, "MPC_GOAL_FAST_SPEED_SCALE", 1.25))
                brake_scale = float(getattr(Config, "MPC_GOAL_BRAKE_SPEED_SCALE", 0.28))
            goal_speed = torch.clamp(speed_base + speed_gain * dist, min=min_speed, max=speed_cap)
            goal_speed = torch.where(dist > fast_dist, goal_speed * fast_scale, goal_speed)
            goal_speed = torch.where(dist < brake_dist, goal_speed * brake_scale, goal_speed)
            goal_approach_mask = goal_nav_mask & (dist > precise_dist)
            ref_vel[goal_approach_mask] = direction[goal_approach_mask] * goal_speed[goal_approach_mask].unsqueeze(-1)

        if getattr(Config, "MPC_WP_Z_SYNC_ENABLE", True):
            z_speed_max = float(getattr(Config, "MPC_WP_Z_REF_SPEED_MAX", 0.85))
            ref_vel[:, 2] = torch.clamp(ref_vel[:, 2], -z_speed_max, z_speed_max)

            near_xy = float(getattr(Config, "MPC_WP_Z_NEAR_XY_DIST", 0.25))
            near_z = float(getattr(Config, "MPC_WP_Z_NEAR_ERR", 0.12))
            xy_scale = float(getattr(Config, "MPC_WP_Z_NEAR_XY_SPEED_SCALE", 0.35))
            z_lag_check = active if obs_count <= 2 else approach_mask
            z_lag_mask = z_lag_check & (dist_xy < near_xy) & (torch.abs(target_rpos[:, 2]) > near_z)
            ref_vel[z_lag_mask, :2] *= xy_scale
        return ref_vel, danger_wp_brake_mask

    def _blend_next_waypoint_direction(
        self,
        parsed,
        current_dir: torch.Tensor,
        target_rpos: torch.Tensor,
        target_slot: torch.Tensor,
        dist: torch.Tensor,
        wp_mask: torch.Tensor,
    ):
        blended = current_dir.clone()
        preview_dist = float(getattr(Config, "MPC_WP_PREVIEW_DIST", 0.55))
        blend_max = float(getattr(Config, "MPC_WP_PREVIEW_BLEND_MAX", 0.28))
        min_next_dist = float(getattr(Config, "MPC_WP_PREVIEW_MIN_NEXT_DIST", 0.20))

        preview_mask = wp_mask & (dist < preview_dist) & (dist > 0.05)
        for b in preview_mask.nonzero(as_tuple=False).view(-1):
            b_idx = int(b.item())
            pending = parsed.waypoint_active[b_idx] & (~parsed.waypoint_visited[b_idx])
            current_slot = int(target_slot[b_idx].item())
            if 0 <= current_slot < pending.shape[0]:
                pending[current_slot] = False
            pending_idx = pending.nonzero(as_tuple=False).view(-1)
            if pending_idx.numel() == 0:
                continue

            current_target_abs = target_rpos[b_idx]
            next_rel_from_drone = parsed.waypoint_rpos[b_idx, pending_idx]
            next_rel_from_target = next_rel_from_drone - current_target_abs.unsqueeze(0)
            next_dist = torch.norm(next_rel_from_target, dim=-1)
            valid = next_dist > min_next_dist
            if not valid.any():
                continue

            valid_idx = pending_idx[valid]
            valid_vec = next_rel_from_target[valid]
            valid_dist = next_dist[valid]
            next_vec = valid_vec[torch.argmin(valid_dist)]
            next_dir = next_vec / torch.norm(next_vec).clamp_min(1.0e-6)

            blend = blend_max * torch.clamp((preview_dist - dist[b_idx]) / max(preview_dist, 1.0e-6), 0.0, 1.0)
            mixed = current_dir[b_idx] * (1.0 - blend) + next_dir * blend
            blended[b_idx] = mixed / torch.norm(mixed).clamp_min(1.0e-6)

        return blended

    def _apply_startup_guard(self, target_rpos: torch.Tensor, hover_mask: torch.Tensor):
        target_guarded = target_rpos.clone()

        xy_clip = float(getattr(Config, "MPC_TARGET_XY_CLIP", float("inf")))
        z_clip = float(getattr(Config, "MPC_TARGET_Z_CLIP", float("inf")))
        if np.isfinite(xy_clip):
            target_guarded[:, :2] = torch.clamp(target_guarded[:, :2], -xy_clip, xy_clip)
        if np.isfinite(z_clip):
            target_guarded[:, 2] = torch.clamp(target_guarded[:, 2], -z_clip, z_clip)

        xy_scale = torch.ones(target_guarded.shape[0], device=target_guarded.device, dtype=target_guarded.dtype)
        z_scale = torch.ones_like(xy_scale)
        high_z_mask = torch.zeros_like(hover_mask, dtype=torch.bool)
        return target_guarded, xy_scale, z_scale, high_z_mask

    def _build_takeoff_profile(self, parsed, hover_mask: torch.Tensor):
        current_up = torch.clamp(-parsed.start_rpos[:, 2], min=0.0)
        nav_mask = ~hover_mask

        startup_frames = max(int(getattr(Config, "MPC_TAKEOFF_MIN_FRAMES", 1)), 1)
        strong_alt = float(getattr(Config, "MPC_TAKEOFF_STRONG_ALT", 0.10))
        release_alt = max(float(getattr(Config, "MPC_TAKEOFF_RELEASE_ALT", 0.24)), strong_alt + 1.0e-3)

        startup_progress = torch.clamp(
            (self._startup_counter.float() + 1.0) / float(startup_frames),
            0.0,
            1.0,
        )
        alt_progress = torch.clamp((current_up - strong_alt) / (release_alt - strong_alt), 0.0, 1.0)
        takeoff_progress = torch.minimum(startup_progress, alt_progress)
        takeoff_progress = torch.where(nav_mask, takeoff_progress, torch.ones_like(takeoff_progress))

        strong_takeoff_mask = nav_mask & (takeoff_progress < 0.35)
        low_alt_mask = nav_mask & (takeoff_progress < 0.999)

        strong_xy_limit = float(getattr(Config, "MPC_TAKEOFF_XY_ACCEL_LIMIT_STRONG", 0.22))
        soft_xy_limit = float(getattr(Config, "MPC_TAKEOFF_XY_ACCEL_LIMIT", 0.40))
        xy_accel_limit = strong_xy_limit + (soft_xy_limit - strong_xy_limit) * takeoff_progress
        xy_accel_limit = torch.where(
            nav_mask,
            xy_accel_limit,
            torch.full_like(xy_accel_limit, float(Config.MPC_A_MAX_XY)),
        )

        strong_tilt = float(getattr(Config, "MPC_TAKEOFF_STRONG_TILT_RAD", 0.035))
        soft_tilt = float(getattr(Config, "MPC_TAKEOFF_MAX_TILT_RAD", 0.05))
        tilt_limit = strong_tilt + (soft_tilt - strong_tilt) * takeoff_progress
        tilt_limit = torch.where(
            nav_mask,
            tilt_limit,
            torch.full_like(tilt_limit, float(Config.MPC_MAX_ROLL_RAD)),
        )
        tilt_limit = torch.clamp(tilt_limit, 0.0, float(Config.MPC_MAX_ROLL_RAD))
        return low_alt_mask, strong_takeoff_mask, current_up, xy_accel_limit, tilt_limit, takeoff_progress

    @staticmethod
    def _wrap_angle(angle: torch.Tensor):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _build_yaw_align_mask(self, yaw: torch.Tensor, hover_mask: torch.Tensor):
        if not getattr(Config, "MPC_YAW_ALIGN_ENABLE", True):
            return torch.zeros_like(hover_mask, dtype=torch.bool), torch.zeros_like(yaw)

        yaw_error = self._wrap_angle(-yaw)
        aligned = torch.abs(yaw_error) < float(getattr(Config, "MPC_YAW_ALIGN_THRESH", 0.05))
        self._yaw_align_hold_counter[aligned] += 1
        self._yaw_align_hold_counter[~aligned] = 0

        min_frames = max(int(getattr(Config, "MPC_YAW_ALIGN_MIN_FRAMES", 45)), 1)
        max_frames = max(int(getattr(Config, "MPC_YAW_ALIGN_MAX_FRAMES", 90)), min_frames)
        hold_frames = max(int(getattr(Config, "MPC_YAW_ALIGN_HOLD_FRAMES", 8)), 1)

        need_min_time = self._startup_counter < min_frames
        need_alignment = (self._startup_counter < max_frames) & (self._yaw_align_hold_counter < hold_frames)
        align_mask = (~hover_mask) & (need_min_time | need_alignment)
        return align_mask, yaw_error

    def _apply_yaw_align_profile(
        self,
        target_rpos: torch.Tensor,
        current_up: torch.Tensor,
        xy_accel_limit: torch.Tensor,
        tilt_limit: torch.Tensor,
        align_mask: torch.Tensor,
    ):
        if not align_mask.any():
            return target_rpos, xy_accel_limit, tilt_limit

        target_aligned = target_rpos.clone()
        desired_up = float(getattr(Config, "MPC_YAW_ALIGN_ALT", 0.30))
        target_aligned[align_mask, 0] = 0.0
        target_aligned[align_mask, 1] = 0.0
        target_aligned[align_mask, 2] = desired_up - current_up[align_mask]

        xy_limit = torch.full_like(xy_accel_limit, float(getattr(Config, "MPC_YAW_ALIGN_XY_ACCEL_LIMIT", 0.08)))
        xy_accel_limit = torch.where(align_mask, xy_limit, xy_accel_limit)

        align_tilt = torch.full_like(tilt_limit, float(getattr(Config, "MPC_YAW_ALIGN_TILT_RAD", 0.020)))
        tilt_limit = torch.where(align_mask, align_tilt, tilt_limit)

        self._prev_u[align_mask, :2] = 0.0
        return target_aligned, xy_accel_limit, tilt_limit

    @staticmethod
    def _drone_xy_from_start(parsed):
        start_xy = torch.tensor(
            [0.5, 2.5],
            dtype=parsed.start_rpos.dtype,
            device=parsed.device,
        )
        return start_xy.unsqueeze(0) - parsed.start_rpos[:, :2]

    def _score_detour_candidate(
        self,
        candidate_xy: torch.Tensor,
        target_xy: torch.Tensor,
        obstacle_xy: torch.Tensor,
        obstacle_radius: torch.Tensor,
        active_mask: torch.Tensor,
        chosen_idx: int,
        drone_xy: torch.Tensor = None,
        arena_min: torch.Tensor = None,
        arena_max: torch.Tensor = None,
    ):
        score = torch.norm(candidate_xy - target_xy).item()
        for i in active_mask.nonzero(as_tuple=False).view(-1):
            idx = int(i.item())
            dist = torch.norm(candidate_xy - obstacle_xy[idx]).item()
            clearance = float(obstacle_radius[idx].item()) + float(getattr(Config, "OBSTACLE_BASE_MARGIN", 0.26))
            if idx == chosen_idx:
                clearance += 0.08
            if dist < clearance:
                score += (clearance - dist) * 8.0

        # Wall-aware: penalize candidates near arena boundaries.
        if drone_xy is not None and arena_min is not None and arena_max is not None:
            wall_margin = float(getattr(Config, "OBSTACLE_WALL_MARGIN", 0.32))
            wall_penalty_gain = float(getattr(Config, "OBSTACLE_DETOUR_WALL_PENALTY", 15.0))
            candidate_abs = drone_xy + candidate_xy
            for ax in range(2):
                dist_min = float(candidate_abs[ax].item()) - float(arena_min[ax].item())
                dist_max = float(arena_max[ax].item()) - float(candidate_abs[ax].item())
                if dist_min < wall_margin:
                    score += wall_penalty_gain * (wall_margin - dist_min)
                if dist_max < wall_margin:
                    score += wall_penalty_gain * (wall_margin - dist_max)
        return score

    def _score_detour_candidate_dense(
        self,
        candidate_xy: torch.Tensor,
        target_xy: torch.Tensor,
        obstacle_xy: torch.Tensor,
        obstacle_radius: torch.Tensor,
        active_mask: torch.Tensor,
        chosen_idx: int,
    ):
        """Dense-obstacle (3+) / 456 detour scoring with segment-clearance penalties."""
        score = torch.norm(candidate_xy - target_xy).item()
        active_obstacle_xy = obstacle_xy[active_mask]
        active_obstacle_radius = obstacle_radius[active_mask]
        if self._obs_count >= 4:
            clearance_limit = self._strategy.detour_clearance
        else:
            clearance_limit = float(getattr(Config, "MPC_SEGMENT_DETOUR_CLEARANCE", 0.35))
        if active_obstacle_xy.numel() > 0:
            to_candidate_clearance = self._segment_clearance_to_targets(
                active_obstacle_xy,
                active_obstacle_radius,
                candidate_xy.unsqueeze(0),
            )[0].item()
            from_candidate_clearance = self._segment_clearance_to_targets(
                active_obstacle_xy - candidate_xy.unsqueeze(0),
                active_obstacle_radius,
                (target_xy - candidate_xy).unsqueeze(0),
            )[0].item()
            score += max(clearance_limit - to_candidate_clearance, 0.0) * 18.0
            score += max(clearance_limit - from_candidate_clearance, 0.0) * 14.0

        for i in active_mask.nonzero(as_tuple=False).view(-1):
            idx = int(i.item())
            dist = torch.norm(candidate_xy - obstacle_xy[idx]).item()
            clearance = float(obstacle_radius[idx].item()) + float(getattr(Config, "OBSTACLE_BASE_MARGIN", 0.26))
            if idx == chosen_idx:
                if self._obs_count >= 4:
                    clearance += self._strategy.detour_extra_margin
                else:
                    clearance += float(getattr(Config, "MPC_SEGMENT_DETOUR_EXTRA_MARGIN", 0.10))
            if dist < clearance:
                score += (clearance - dist) * 14.0
        return score

    def _build_obstacle_avoidance_target(
        self,
        parsed,
        target_rpos: torch.Tensor,
        hover_mask: torch.Tensor,
        yaw_align_mask: torch.Tensor,
        v_world: torch.Tensor,
    ):
        avoid_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        avoid_slot = torch.full((parsed.batch_size,), -1, dtype=torch.long, device=parsed.device)
        avoid_side = torch.zeros(parsed.batch_size, dtype=torch.float32, device=parsed.device)
        if not getattr(Config, "OBSTACLE_AVOID_ENABLE", True):
            return target_rpos, avoid_mask, avoid_slot, avoid_side

        target_avoid = target_rpos.clone()
        drone_xy = self._drone_xy_from_start(parsed)
        wall_margin = float(getattr(Config, "OBSTACLE_WALL_MARGIN", 0.32))
        arena_min = torch.tensor(
            [float(Config.ARENA_X_MIN) + wall_margin, float(Config.ARENA_Y_MIN) + wall_margin],
            dtype=target_rpos.dtype, device=parsed.device,
        )
        arena_max = torch.tensor(
            [float(Config.ARENA_X_MAX) - wall_margin, float(Config.ARENA_Y_MAX) - wall_margin],
            dtype=target_rpos.dtype, device=parsed.device,
        )

        lookahead_min = float(getattr(Config, "OBSTACLE_LOOKAHEAD_MIN", 0.05))
        normal_trigger_min = float(getattr(Config, "OBSTACLE_NORMAL_TRIGGER_MIN", 0.80))
        detour_margin = float(getattr(Config, "OBSTACLE_DETOUR_MARGIN", 0.42))
        detour_advance = float(getattr(Config, "OBSTACLE_DETOUR_ADVANCE", 0.45))

        obs_count = self._obs_count

        # Sparse-only config values
        lookahead_max_cfg = float(getattr(Config, "OBSTACLE_LOOKAHEAD_MAX", 1.35))
        speed_margin_gain = float(getattr(Config, "OBSTACLE_SPEED_MARGIN_GAIN", 0.14))
        base_margin = float(getattr(Config, "OBSTACLE_BASE_MARGIN", 0.26))
        hold_frames = max(int(getattr(Config, "OBSTACLE_AVOID_HOLD_FRAMES", 30)), 1)
        transition_enable = getattr(Config, "OBSTACLE_AVOID_TRANSITION_ENABLE", True)
        transition_blend = float(getattr(Config, "OBSTACLE_AVOID_TRANSITION_BLEND", 0.55))
        transition_dist = float(getattr(Config, "OBSTACLE_AVOID_TRANSITION_DIST", 0.70))
        hold_frames_max = max(int(getattr(Config, "OBSTACLE_AVOID_HOLD_FRAMES_MAX", 60)), hold_frames)

        # Dense / 456 config values
        if obs_count >= 4:
            lookahead_max_cfg_dense = self._strategy.obs_lookahead_max
            speed_margin_gain_dense = self._strategy.obs_speed_margin_gain
            hold_frames_dense = self._strategy.obs_avoid_hold_frames
            segment_hold_frames = max(int(getattr(Config, "MPC_SEGMENT_DETOUR_HOLD_FRAMES", 28)), hold_frames_dense)
            base_margin = self._strategy.obs_base_margin
            detour_margin = self._strategy.obs_detour_margin
            detour_advance = self._strategy.obs_detour_advance
            wall_margin = self._strategy.obs_wall_margin
            normal_trigger_min = self._strategy.obs_normal_trigger_min
        elif obs_count == 3:
            lookahead_max_cfg_dense = float(getattr(Config, "OBSTACLE_LOOKAHEAD_MAX_DENSE", 1.00))
            speed_margin_gain_dense = float(getattr(Config, "OBSTACLE_SPEED_MARGIN_GAIN_DENSE", 0.08))
            hold_frames_dense = max(int(getattr(Config, "OBSTACLE_AVOID_HOLD_FRAMES_DENSE", 18)), 1)
            segment_hold_frames = max(int(getattr(Config, "MPC_SEGMENT_DETOUR_HOLD_FRAMES", 28)), hold_frames_dense)

        for b in range(parsed.batch_size):
            if hover_mask[b] or yaw_align_mask[b]:
                self._avoid_obstacle_slot[b] = -1
                self._avoid_hold_counter[b] = 0
                if obs_count <= 2:
                    self._avoid_blend_weight[b] = 0.0
                    self._avoid_hold_total[b] = 0
                continue

            target_xy = target_rpos[b, :2]
            target_dist = torch.norm(target_xy).item()
            if target_dist < 0.15:
                self._avoid_obstacle_slot[b] = -1
                self._avoid_hold_counter[b] = 0
                if obs_count <= 2:
                    self._avoid_blend_weight[b] = 0.0
                    self._avoid_hold_total[b] = 0
                continue

            direction = target_xy / max(target_dist, 1.0e-6)
            normal = torch.stack((-direction[1], direction[0]))
            obstacle_xy = parsed.obstacle_rpos[b, :, :2]
            obstacle_radius = parsed.obstacle_radius[b]
            active_mask = parsed.obstacle_active[b]
            speed_xy = torch.norm(v_world[b, :2]).item()

            is_dense = obs_count >= 3

            if is_dense:
                # === Dense path (3+ obstacles): segment detour ===
                lh_max = min(lookahead_max_cfg_dense + 0.20 * speed_xy, max(target_dist - 0.05, lookahead_min))
                sg = speed_margin_gain_dense
                hf = hold_frames_dense
                post_danger_guard_active = int(self._post_danger_guard_counter[b].item()) > 0

                best_idx = -1
                best_score = float("inf")
                best_from_segment = False
                for i in active_mask.nonzero(as_tuple=False).view(-1):
                    idx = int(i.item())
                    obs_xy = obstacle_xy[idx]
                    forward = torch.dot(obs_xy, direction).item()
                    if forward < lookahead_min or forward > lh_max:
                        continue
                    lateral_vec = obs_xy - direction * forward
                    lateral = torch.norm(lateral_vec).item()
                    radius = float(obstacle_radius[idx].item())
                    clearance = radius + base_margin + sg * speed_xy
                    if lateral >= clearance:
                        continue
                    normal_zone_bonus = 0.0 if forward >= normal_trigger_min else 0.15
                    penetration = max(clearance - lateral, 0.0)
                    score = forward - 0.45 * penetration + normal_zone_bonus
                    if score < best_score:
                        best_score = score
                        best_idx = idx

                if (best_idx < 0 or post_danger_guard_active) and getattr(Config, "MPC_SEGMENT_DETOUR_ENABLE", True):
                    segment_clearance = float(getattr(Config, "MPC_SEGMENT_DETOUR_CLEARANCE", 0.35))
                    if post_danger_guard_active:
                        segment_clearance = max(
                            segment_clearance,
                            float(getattr(Config, "MPC_POST_DANGER_SEGMENT_CLEARANCE", segment_clearance)),
                        )
                    segment_best_idx = -1
                    segment_best_score = float("inf")
                    for i in active_mask.nonzero(as_tuple=False).view(-1):
                        idx = int(i.item())
                        obs_xy = obstacle_xy[idx]
                        forward = torch.dot(obs_xy, direction).item()
                        if forward < lookahead_min or forward > target_dist - 0.05:
                            continue
                        lateral_vec = obs_xy - direction * forward
                        lateral = torch.norm(lateral_vec).item()
                        radius = float(obstacle_radius[idx].item())
                        line_clearance = lateral - radius
                        if line_clearance >= segment_clearance:
                            continue
                        penetration = max(segment_clearance - line_clearance, 0.0)
                        score = forward - 0.35 * penetration
                        if score < segment_best_score:
                            segment_best_score = score
                            segment_best_idx = idx
                    if segment_best_idx >= 0:
                        best_score = segment_best_score
                        best_idx = segment_best_idx
                        best_from_segment = True

                if best_idx < 0:
                    self._avoid_obstacle_slot[b] = -1
                    self._avoid_hold_counter[b] = 0
                    continue

                radius = float(obstacle_radius[best_idx].item())
                extra_margin = float(getattr(Config, "MPC_SEGMENT_DETOUR_EXTRA_MARGIN", 0.10)) if best_from_segment else 0.0
                detour_offset = radius + detour_margin + extra_margin + sg * speed_xy
                advance_cfg = float(getattr(Config, "MPC_SEGMENT_DETOUR_ADVANCE", detour_advance)) if best_from_segment else detour_advance
                advance = min(advance_cfg + radius * 0.5, 0.85)
                obs_xy_chosen = obstacle_xy[best_idx]
                active_hold_frames = segment_hold_frames if best_from_segment else hf

                reuse_side = (
                    int(self._avoid_obstacle_slot[b].item()) == best_idx
                    and int(self._avoid_hold_counter[b].item()) < active_hold_frames
                )
                candidate_sides = [float(self._avoid_side[b].item())] if reuse_side else [1.0, -1.0]
                radius_scales = getattr(Config, "MPC_SEGMENT_DETOUR_RADIUS_SCALES", [1.0])
                if not best_from_segment:
                    radius_scales = [1.0]
                best_side = candidate_sides[0]
                best_candidate = None
                best_candidate_score = float("inf")
                for side in candidate_sides:
                    for scale in radius_scales:
                        candidate_rel = obs_xy_chosen + normal * (side * detour_offset * float(scale)) + direction * advance
                        candidate_abs = torch.clamp(drone_xy[b] + candidate_rel, arena_min, arena_max)
                        candidate_rel = candidate_abs - drone_xy[b]
                        candidate_score = self._score_detour_candidate_dense(
                            candidate_rel, target_xy, obstacle_xy, obstacle_radius,
                            active_mask, best_idx,
                        )
                        if candidate_score < best_candidate_score:
                            best_candidate_score = candidate_score
                            best_candidate = candidate_rel
                            best_side = side

                target_avoid[b, :2] = best_candidate
                avoid_mask[b] = True
                avoid_slot[b] = best_idx
                avoid_side[b] = best_side
                self._avoid_obstacle_slot[b] = best_idx
                self._avoid_side[b] = best_side
                self._avoid_hold_counter[b] = min(int(self._avoid_hold_counter[b].item()) + 1, active_hold_frames)

            else:
                # === Sparse path (0/1/2 obstacles): transition blending + wall trap ===
                lh_max = min(lookahead_max_cfg + 0.20 * speed_xy, max(target_dist - 0.05, lookahead_min))
                sg = speed_margin_gain
                hf = hold_frames

                best_idx = -1
                best_score = float("inf")
                for i in active_mask.nonzero(as_tuple=False).view(-1):
                    idx = int(i.item())
                    obs_xy = obstacle_xy[idx]
                    forward = torch.dot(obs_xy, direction).item()
                    if forward < lookahead_min or forward > lh_max:
                        continue
                    lateral_vec = obs_xy - direction * forward
                    lateral = torch.norm(lateral_vec).item()
                    radius = float(obstacle_radius[idx].item())
                    clearance = radius + base_margin + sg * speed_xy
                    if lateral >= clearance:
                        continue
                    normal_zone_bonus = 0.0 if forward >= normal_trigger_min else 0.15
                    penetration = max(clearance - lateral, 0.0)
                    score = forward - 0.45 * penetration + normal_zone_bonus
                    if score < best_score:
                        best_score = score
                        best_idx = idx

                if best_idx < 0:
                    self._avoid_obstacle_slot[b] = -1
                    self._avoid_hold_counter[b] = 0
                    self._avoid_blend_weight[b] = 0.0
                    self._avoid_hold_total[b] = 0
                    continue

                radius = float(obstacle_radius[best_idx].item())
                detour_offset = radius + detour_margin + sg * speed_xy
                advance = min(detour_advance + radius * 0.5, 0.75)
                obs_xy_chosen = obstacle_xy[best_idx]

                reuse_side = (
                    int(self._avoid_obstacle_slot[b].item()) == best_idx
                    and int(self._avoid_hold_counter[b].item()) < hf
                )
                candidate_sides = [float(self._avoid_side[b].item())] if reuse_side else [1.0, -1.0]
                best_side = candidate_sides[0]
                best_candidate = None
                best_candidate_score = float("inf")
                for side in candidate_sides:
                    candidate_rel = obs_xy_chosen + normal * (side * detour_offset) + direction * advance
                    candidate_abs = torch.clamp(drone_xy[b] + candidate_rel, arena_min, arena_max)
                    candidate_rel = candidate_abs - drone_xy[b]
                    candidate_score = self._score_detour_candidate(
                        candidate_rel, target_xy, obstacle_xy, obstacle_radius,
                        active_mask, best_idx,
                        drone_xy[b], arena_min, arena_max,
                    )
                    if candidate_score < best_candidate_score:
                        best_candidate_score = candidate_score
                        best_candidate = candidate_rel
                        best_side = side

                # Wall-trap detection
                if best_candidate is not None:
                    best_abs = drone_xy[b] + best_candidate
                    near_wall = False
                    for ax in range(2):
                        d_min = float(best_abs[ax].item()) - float(arena_min[ax].item())
                        d_max = float(arena_max[ax].item()) - float(best_abs[ax].item())
                        if d_min < wall_margin * 0.6 or d_max < wall_margin * 0.6:
                            near_wall = True
                            break
                    if near_wall and len(candidate_sides) == 1:
                        opp_side = -best_side
                        opp_rel = obs_xy_chosen + normal * (opp_side * detour_offset) + direction * advance
                        opp_abs = torch.clamp(drone_xy[b] + opp_rel, arena_min, arena_max)
                        opp_rel = opp_abs - drone_xy[b]
                        opp_score = self._score_detour_candidate(
                            opp_rel, target_xy, obstacle_xy, obstacle_radius,
                            active_mask, best_idx,
                            drone_xy[b], arena_min, arena_max,
                        )
                        if opp_score < best_candidate_score:
                            best_candidate_score = opp_score
                            best_candidate = opp_rel
                            best_side = opp_side
                            self._avoid_side[b] = opp_side

                target_avoid[b, :2] = best_candidate
                avoid_mask[b] = True
                avoid_slot[b] = best_idx
                avoid_side[b] = best_side
                self._avoid_obstacle_slot[b] = best_idx
                self._avoid_side[b] = best_side

                # Attraction-source transition blending
                blend_weight = 0.0
                dir_alignment = 0.0
                obs_norm = torch.norm(obs_xy_chosen).item()
                if transition_enable and obs_norm > 1.0e-4 and target_dist > 1.0e-4:
                    obs_dir_xy = obs_xy_chosen / obs_norm
                    target_dir_xy = target_xy / target_dist
                    dir_alignment = max(torch.dot(obs_dir_xy, target_dir_xy).item(), 0.0)
                    raw_blend = dir_alignment * (1.0 - min(obs_norm / max(transition_dist, 1.0e-4), 1.0))
                    blend_weight = min(raw_blend * transition_blend, 0.95)
                    detour_to_target = target_xy - best_candidate
                    dt_dist = torch.norm(detour_to_target).item()
                    if dt_dist > 0.05:
                        clearance_to_detour = self._segment_clearance_to_targets(
                            obstacle_xy[active_mask], obstacle_radius[active_mask],
                            detour_to_target.unsqueeze(0),
                        )
                        if clearance_to_detour.numel() > 0 and clearance_to_detour[0].item() < base_margin:
                            blend_weight = max(blend_weight, transition_blend)
                    target_avoid[b, :2] = (
                        best_candidate * blend_weight + target_xy * (1.0 - blend_weight)
                    )
                    block_still = dir_alignment > 0.25
                    hold_limit = hold_frames_max if block_still else hf
                else:
                    hold_limit = hf

                self._avoid_hold_counter[b] = min(int(self._avoid_hold_counter[b].item()) + 1, hold_limit)
                self._avoid_blend_weight[b] = blend_weight

        return target_avoid, avoid_mask, avoid_slot, avoid_side

    def _apply_obstacle_safety_accel(
        self,
        parsed,
        accel_cmd: torch.Tensor,
        hover_mask: torch.Tensor,
        yaw_align_mask: torch.Tensor,
        v_world: torch.Tensor,
    ):
        emergency_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        tangential_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        wall_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        if not getattr(Config, "OBSTACLE_AVOID_ENABLE", True):
            return accel_cmd, emergency_mask, tangential_mask, wall_mask

        # Sparse config
        margin = float(getattr(Config, "OBSTACLE_EMERGENCY_MARGIN", 0.40))
        gain = float(getattr(Config, "OBSTACLE_EMERGENCY_GAIN", 2.2))
        accel_max = float(getattr(Config, "OBSTACLE_EMERGENCY_ACCEL_MAX", 1.8))
        tan_gain = float(getattr(Config, "OBSTACLE_TANGENTIAL_GAIN", 0.65))
        tan_max = float(getattr(Config, "OBSTACLE_TANGENTIAL_MAX", 1.2))
        lock_enable = getattr(Config, "TAN_DIR_LOCK_ENABLE", True)
        boundary_enable = getattr(Config, "BOUNDARY_REPULSION_ENABLE", True)
        wall_margin_bp = float(getattr(Config, "BOUNDARY_REPULSION_MARGIN", 0.35))
        wall_gain = float(getattr(Config, "BOUNDARY_REPULSION_GAIN", 2.5))

        # Dense / 456 config
        obs_count = self._obs_count
        if obs_count >= 4:
            margin_dense = self._strategy.emergency_margin
        elif obs_count == 3:
            margin_dense = float(getattr(Config, "OBSTACLE_EMERGENCY_MARGIN_DENSE", 0.24))
        else:
            margin_dense = margin

        adjusted = accel_cmd.clone()

        for b in range(parsed.batch_size):
            if hover_mask[b] or yaw_align_mask[b]:
                self._tan_direction_lock[b] = 0.0
                self._tan_lock_counter[b] = 0
                continue

            is_dense = obs_count >= 3

            if is_dense:
                # === Dense path: simple radial repulsion (hard safety handles the rest) ===
                margin_use = margin_dense
                repel = torch.zeros(2, dtype=accel_cmd.dtype, device=accel_cmd.device)
                for i in parsed.obstacle_active[b].nonzero(as_tuple=False).view(-1):
                    idx = int(i.item())
                    obs_xy = parsed.obstacle_rpos[b, idx, :2]
                    dist = torch.norm(obs_xy).item()
                    radius = float(parsed.obstacle_radius[b, idx].item())
                    safe_dist = radius + margin_use
                    if dist >= safe_dist or dist < 1.0e-4:
                        continue
                    away = -obs_xy / max(dist, 1.0e-4)
                    closing_speed = max(torch.dot(v_world[b, :2], obs_xy / max(dist, 1.0e-4)).item(), 0.0)
                    strength = gain * (safe_dist - dist) / safe_dist + 0.35 * closing_speed
                    repel += away * strength
                    emergency_mask[b] = True

                repel_norm = torch.norm(repel).item()
                if repel_norm > accel_max:
                    repel = repel * (accel_max / max(repel_norm, 1.0e-6))
                adjusted[b, :2] += repel

            else:
                # === Sparse path: full tangential + boundary + trap + z-climb ===
                margin_use = margin

                goal_xy = parsed.goal_rpos[b, :2]
                goal_dist = torch.norm(goal_xy).item()
                goal_dir = goal_xy / max(goal_dist, 1.0e-6) if goal_dist > 1.0e-4 else torch.zeros(2, device=parsed.device)
                goal_perp = torch.stack([-goal_dir[1], goal_dir[0]])

                # Tangential direction lock pre-scan
                any_threat = False
                net_tan_proj = 0.0
                if lock_enable and tan_gain > 0.0 and goal_dist > 1.0e-4:
                    for i in parsed.obstacle_active[b].nonzero(as_tuple=False).view(-1):
                        idx = int(i.item())
                        obs_xy = parsed.obstacle_rpos[b, idx, :2]
                        dist = torch.norm(obs_xy).item()
                        radius = float(parsed.obstacle_radius[b, idx].item())
                        safe_dist = radius + margin_use
                        if dist >= safe_dist or dist < 1.0e-4:
                            continue
                        any_threat = True
                        away = -obs_xy / max(dist, 1.0e-4)
                        tan_this = torch.stack([-away[1], away[0]])
                        proj = float(torch.dot(tan_this, goal_perp).item())
                        weight = gain * (safe_dist - dist) / max(safe_dist, 1.0e-4)
                        net_tan_proj += proj * weight

                    if self._tan_direction_lock[b] == 0.0 and any_threat:
                        self._tan_direction_lock[b] = 1.0 if net_tan_proj > 0.001 else -1.0
                        self._tan_lock_counter[b] = 0
                    elif not any_threat:
                        self._tan_direction_lock[b] = 0.0
                        self._tan_lock_counter[b] = 0
                    elif self._tan_direction_lock[b] != 0.0:
                        self._tan_lock_counter[b] += 1

                use_locked = lock_enable and self._tan_direction_lock[b] != 0.0 and goal_dist > 1.0e-4
                locked_tan_dir = goal_perp * self._tan_direction_lock[b] if use_locked else None

                repel = torch.zeros(2, dtype=accel_cmd.dtype, device=accel_cmd.device)
                for i in parsed.obstacle_active[b].nonzero(as_tuple=False).view(-1):
                    idx = int(i.item())
                    obs_xy = parsed.obstacle_rpos[b, idx, :2]
                    dist = torch.norm(obs_xy).item()
                    radius = float(parsed.obstacle_radius[b, idx].item())
                    safe_dist = radius + margin_use
                    if dist >= safe_dist or dist < 1.0e-4:
                        continue
                    away = -obs_xy / max(dist, 1.0e-4)
                    closing_speed = max(torch.dot(v_world[b, :2], obs_xy / max(dist, 1.0e-4)).item(), 0.0)
                    strength = gain * (safe_dist - dist) / safe_dist + 0.35 * closing_speed
                    repel += away * strength
                    emergency_mask[b] = True

                    if tan_gain > 0.0 and goal_dist > 1.0e-4:
                        if locked_tan_dir is not None:
                            tan_dir = locked_tan_dir
                        else:
                            tan_dir = torch.stack([-away[1], away[0]])
                            proj_goal = torch.dot(tan_dir, goal_dir).item()
                            if proj_goal < 0.0:
                                tan_dir = -tan_dir
                        alignment = abs(torch.dot(away, goal_dir).item())
                        tan_strength = tan_gain * alignment * (safe_dist - dist) / max(safe_dist, 1.0e-4)
                        tan_strength = min(tan_strength, tan_max)
                        repel += tan_dir * tan_strength
                        tangential_mask[b] = True

                # Arena boundary repulsion
                if boundary_enable:
                    drone_abs_xy = self._drone_xy_from_start(parsed)[b]
                    x_min = float(Config.ARENA_X_MIN)
                    x_max = float(Config.ARENA_X_MAX)
                    y_min = float(Config.ARENA_Y_MIN)
                    y_max = float(Config.ARENA_Y_MAX)
                    dist_x_min = float(drone_abs_xy[0].item()) - x_min
                    dist_x_max = x_max - float(drone_abs_xy[0].item())
                    dist_y_min = float(drone_abs_xy[1].item()) - y_min
                    dist_y_max = y_max - float(drone_abs_xy[1].item())
                    if dist_x_min < wall_margin_bp:
                        repel[0] += wall_gain * (wall_margin_bp - dist_x_min) / max(wall_margin_bp, 1.0e-4)
                        wall_mask[b] = True
                    if dist_x_max < wall_margin_bp:
                        repel[0] -= wall_gain * (wall_margin_bp - dist_x_max) / max(wall_margin_bp, 1.0e-4)
                        wall_mask[b] = True
                    if dist_y_min < wall_margin_bp:
                        repel[1] += wall_gain * (wall_margin_bp - dist_y_min) / max(wall_margin_bp, 1.0e-4)
                        wall_mask[b] = True
                    if dist_y_max < wall_margin_bp:
                        repel[1] -= wall_gain * (wall_margin_bp - dist_y_max) / max(wall_margin_bp, 1.0e-4)
                        wall_mask[b] = True

                # Trap escape
                if emergency_mask[b] and wall_mask[b]:
                    trap_gain = float(getattr(Config, "OBSTACLE_TRAP_ESCAPE_GAIN", 1.5))
                    trap_force = torch.zeros(2, dtype=accel_cmd.dtype, device=accel_cmd.device)
                    drone_abs_xy = self._drone_xy_from_start(parsed)[b]
                    near_y_wall = (
                        (float(drone_abs_xy[1].item()) - y_min) < wall_margin_bp * 1.2
                        or (y_max - float(drone_abs_xy[1].item())) < wall_margin_bp * 1.2
                    )
                    near_x_wall = (
                        (float(drone_abs_xy[0].item()) - x_min) < wall_margin_bp * 1.2
                        or (x_max - float(drone_abs_xy[0].item())) < wall_margin_bp * 1.2
                    )
                    if near_y_wall and not near_x_wall:
                        trap_force[0] = trap_gain * (1.0 if goal_dir[0].item() > 0 else -1.0)
                    elif near_x_wall and not near_y_wall:
                        trap_force[1] = trap_gain * (1.0 if goal_dir[1].item() > 0 else -1.0)
                    else:
                        trap_force[0] = trap_gain * 0.7 * (1.0 if goal_dir[0].item() > 0 else -1.0)
                        trap_force[1] = trap_gain * 0.7 * (1.0 if goal_dir[1].item() > 0 else -1.0)
                    repel += trap_force

                repel_norm = torch.norm(repel).item()
                if repel_norm > accel_max:
                    repel = repel * (accel_max / max(repel_norm, 1.0e-6))
                adjusted[b, :2] += repel

                if repel_norm > 0.01:
                    z_climb_gain = float(getattr(Config, "OBSTACLE_EMERGENCY_Z_CLIMB_GAIN", 0.25))
                    adjusted[b, 2] += repel_norm * z_climb_gain

        return adjusted, emergency_mask, tangential_mask, wall_mask

    @staticmethod
    def _limit_xy_accel(accel_cmd: torch.Tensor, xy_accel_limit: torch.Tensor):
        xy = accel_cmd[:, :2]
        xy_norm = torch.norm(xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
        scale = torch.clamp(xy_accel_limit.unsqueeze(-1) / xy_norm, max=1.0)
        accel_cmd[:, :2] = xy * scale
        return accel_cmd

    def _apply_accel_slew_limit(self, accel_cmd: torch.Tensor, takeoff_progress: torch.Tensor):
        xy_slew_low = float(getattr(Config, "MPC_TAKEOFF_XY_SLEW_LIMIT", 0.04))
        z_slew_low = float(getattr(Config, "MPC_TAKEOFF_Z_SLEW_LIMIT", 0.08))

        if self._obs_count >= 3:
            xy_slew_high = float(getattr(Config, "MPC_XY_SLEW_LIMIT_DENSE", 0.37125))
            z_slew_high = float(getattr(Config, "MPC_Z_SLEW_LIMIT_DENSE", 0.675))
        else:
            xy_slew_high = float(getattr(Config, "MPC_XY_SLEW_LIMIT", 0.55))
            z_slew_high = float(getattr(Config, "MPC_Z_SLEW_LIMIT", 0.85))

        xy_slew = xy_slew_low + (xy_slew_high - xy_slew_low) * takeoff_progress
        z_slew = z_slew_low + (z_slew_high - z_slew_low) * takeoff_progress

        delta = accel_cmd - self._prev_u
        delta[:, :2] = torch.clamp(delta[:, :2], -xy_slew.unsqueeze(-1), xy_slew.unsqueeze(-1))
        delta[:, 2] = torch.clamp(delta[:, 2], -z_slew, z_slew)
        return self._prev_u + delta

    def _build_goal_hover_mask(self, parsed, hover_mask: torch.Tensor, v_world: torch.Tensor):
        goal_xy = torch.norm(parsed.goal_rpos[:, :2], dim=-1)
        goal_z = torch.abs(parsed.goal_rpos[:, 2])
        speed_xy = torch.norm(v_world[:, :2], dim=-1)

        near_goal_xy = float(getattr(Config, "MPC_GOAL_BIAS_DIST_XY", 0.35))
        near_goal_z = float(getattr(Config, "MPC_GOAL_BIAS_DIST_Z", 0.25))

        if self._obs_count >= 3:
            near_goal_speed = float(getattr(Config, "MPC_GOAL_BIAS_SPEED_XY_DENSE", 0.24))
        else:
            near_goal_speed = float(getattr(Config, "MPC_GOAL_BIAS_SPEED_XY", 0.30))
        return (~hover_mask) & (goal_xy < near_goal_xy) & (goal_z < near_goal_z) & (speed_xy < near_goal_speed)

    def _map_accel_to_action(
        self,
        accel_cmd_world: torch.Tensor,
        rotation_matrix: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        angular_velocity: torch.Tensor,
        v_world: torch.Tensor,
        hover_mask: torch.Tensor,
        near_goal_mask: torch.Tensor,
        low_alt_mask: torch.Tensor,
        tilt_limit: torch.Tensor,
    ):
        g = getattr(Config, "MPC_GRAVITY", 9.81)
        accel_cmd_body = torch.bmm(
            rotation_matrix.transpose(1, 2),
            accel_cmd_world.unsqueeze(-1),
        ).squeeze(-1)

        roll_cmd = torch.clamp(
            accel_cmd_body[:, 1] / g,
            -Config.MPC_MAX_ROLL_RAD,
            Config.MPC_MAX_ROLL_RAD,
        )
        pitch_cmd = torch.clamp(
            -accel_cmd_body[:, 0] / g,
            -Config.MPC_MAX_PITCH_RAD,
            Config.MPC_MAX_PITCH_RAD,
        )

        roll_cmd = torch.maximum(torch.minimum(roll_cmd, tilt_limit), -tilt_limit)
        pitch_cmd = torch.maximum(torch.minimum(pitch_cmd, tilt_limit), -tilt_limit)

        roll_rate = (
            Config.MPC_ATT_KP_ROLL * (roll_cmd - roll)
            - getattr(Config, "MPC_ATT_KD_ROLL", 0.0) * angular_velocity[:, 0]
        )
        pitch_rate = (
            Config.MPC_ATT_KP_PITCH * (pitch_cmd - pitch)
            - getattr(Config, "MPC_ATT_KD_PITCH", 0.0) * angular_velocity[:, 1]
        )
        roll_rate = torch.clamp(roll_rate, -Config.MPC_MAX_ROLL_RATE, Config.MPC_MAX_ROLL_RATE)
        pitch_rate = torch.clamp(pitch_rate, -Config.MPC_MAX_PITCH_RATE, Config.MPC_MAX_PITCH_RATE)

        thrust = Config.HOVER_BASE_THRUST + Config.MPC_THRUST_GAIN * accel_cmd_world[:, 2]
        vz_damp = torch.full_like(thrust, getattr(Config, "MPC_VZ_DAMP", 0.14))
        goal_hover_mask = hover_mask | near_goal_mask
        vz_damp = torch.where(
            goal_hover_mask,
            torch.full_like(vz_damp, getattr(Config, "MPC_GOAL_HOVER_VZ_DAMP", 0.22)),
            vz_damp,
        )
        thrust = thrust - vz_damp * v_world[:, 2]
        tilt_angle = torch.sqrt(roll_cmd * roll_cmd + pitch_cmd * pitch_cmd + 1.0e-8)
        thrust = thrust / torch.clamp(torch.cos(tilt_angle), 0.92, 1.0)
        thrust = torch.clamp(thrust, Config.MPC_THRUST_MIN, Config.MPC_THRUST_MAX)

        action = torch.zeros(accel_cmd_world.shape[0], 4, device=accel_cmd_world.device, dtype=torch.float32)
        action[:, 0] = roll_rate * getattr(Config, "ACTION_ROLL_SIGN", 1.0)
        action[:, 1] = pitch_rate * getattr(Config, "ACTION_PITCH_SIGN", 1.0)
        action[:, 2] = 0.0
        action[:, 3] = thrust
        return action, roll_cmd, pitch_cmd, accel_cmd_body

    # -----------------------------------------------------------------------
    # Dense-obstacle (3+) safety layers
    # -----------------------------------------------------------------------

    def _apply_hard_obstacle_safety(
        self,
        parsed,
        accel_cmd: torch.Tensor,
        hover_mask: torch.Tensor,
        yaw_align_mask: torch.Tensor,
        v_world: torch.Tensor,
    ):
        """Accel-level safety override with hysteresis (dense / 456, replaces Binary Tide)."""
        hard_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        if not getattr(Config, "MPC_HARD_SAFETY_ENABLE", True):
            return accel_cmd, hard_mask

        obs_count = self._obs_count
        if obs_count >= 4:
            trigger_clearance = self._strategy.hard_safety_clearance
            release_clearance = self._strategy.hard_safety_release_clearance
            away_accel = self._strategy.hard_safety_away_accel
            brake_gain = self._strategy.hard_safety_brake_gain
            max_accel = self._strategy.hard_safety_max_accel
        else:
            trigger_clearance = float(getattr(Config, "MPC_HARD_SAFETY_CLEARANCE", 0.18))
            release_clearance = float(getattr(Config, "MPC_HARD_SAFETY_RELEASE_CLEARANCE", 0.26))
            away_accel = float(getattr(Config, "MPC_HARD_SAFETY_AWAY_ACCEL", 2.4))
            brake_gain = float(getattr(Config, "MPC_HARD_SAFETY_BRAKE_GAIN", 1.8))
            max_accel = float(getattr(Config, "MPC_HARD_SAFETY_MAX_ACCEL", 2.6))
        adjusted = accel_cmd.clone()

        for b in range(parsed.batch_size):
            if hover_mask[b] or yaw_align_mask[b]:
                self._hard_safety_counter[b] = 0
                continue

            best_clearance = float("inf")
            best_away = None
            best_closing = 0.0
            for i in parsed.obstacle_active[b].nonzero(as_tuple=False).view(-1):
                idx = int(i.item())
                obs_xy = parsed.obstacle_rpos[b, idx, :2]
                dist = torch.norm(obs_xy).item()
                if dist < 1.0e-4:
                    continue
                clearance = dist - float(parsed.obstacle_radius[b, idx].item())
                obs_dir = obs_xy / max(dist, 1.0e-4)
                closing = torch.dot(v_world[b, :2], obs_dir).item()
                if clearance < best_clearance:
                    best_clearance = clearance
                    best_away = -obs_dir
                    best_closing = closing

            if best_away is None:
                self._hard_safety_counter[b] = 0
                continue

            was_active = int(self._hard_safety_counter[b].item()) > 0
            active = best_clearance < trigger_clearance or (was_active and best_clearance < release_clearance)
            if not active or best_closing <= 0.0:
                self._hard_safety_counter[b] = 0
                continue

            toward_accel = torch.dot(adjusted[b, :2], -best_away)
            if toward_accel.item() > 0.0:
                adjusted[b, :2] = adjusted[b, :2] - toward_accel * (-best_away)

            strength = away_accel * (trigger_clearance - min(best_clearance, trigger_clearance)) / max(trigger_clearance, 1.0e-6)
            strength = strength + brake_gain * best_closing
            adjusted[b, :2] = adjusted[b, :2] + best_away * strength

            accel_norm = torch.norm(adjusted[b, :2]).item()
            if accel_norm > max_accel:
                adjusted[b, :2] = adjusted[b, :2] * (max_accel / max(accel_norm, 1.0e-6))

            self._hard_safety_counter[b] = min(int(self._hard_safety_counter[b].item()) + 1, 1000)
            hard_mask[b] = True

        return adjusted, hard_mask

    def _apply_speed_dependent_target_brake(
        self,
        parsed,
        accel_cmd: torch.Tensor,
        target_rpos: torch.Tensor,
        target_slot: torch.Tensor,
        hover_mask: torch.Tensor,
        yaw_align_mask: torch.Tensor,
        v_world: torch.Tensor,
    ):
        """Deceleration when approaching obstacles too fast (dense / 456)."""
        brake_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        if not getattr(Config, "MPC_SPEED_BRAKE_ENABLE", True):
            return accel_cmd, brake_mask

        obs_count = self._obs_count
        if obs_count >= 4:
            decel = max(self._strategy.brake_decel, 1.0e-3)
            buffer_dist = self._strategy.brake_buffer
            min_brake_dist = self._strategy.brake_dist_min
            gain = self._strategy.brake_gain
            max_accel = self._strategy.brake_max_accel
            near_obs_clearance = self._strategy.brake_near_obs_clearance
            near_obs_speed = self._strategy.brake_near_obs_speed
            safe_speed = self._strategy.brake_safe_speed
        else:
            decel = max(float(getattr(Config, "MPC_SPEED_BRAKE_DECEL", 1.8)), 1.0e-3)
            buffer_dist = float(getattr(Config, "MPC_SPEED_BRAKE_BUFFER", 0.18))
            min_brake_dist = float(getattr(Config, "MPC_SPEED_BRAKE_DIST_MIN", 0.35))
            gain = float(getattr(Config, "MPC_SPEED_BRAKE_GAIN", 1.35))
            max_accel = float(getattr(Config, "MPC_SPEED_BRAKE_MAX_ACCEL", 1.8))
            near_obs_clearance = float(getattr(Config, "MPC_SPEED_BRAKE_NEAR_OBS_CLEARANCE", 0.65))
            near_obs_speed = float(getattr(Config, "MPC_SPEED_BRAKE_NEAR_OBS_SPEED", 0.25))
            safe_speed = float(getattr(Config, "MPC_SPEED_BRAKE_SAFE_SPEED", 0.55))
        adjusted = accel_cmd.clone()

        for b in range(parsed.batch_size):
            slot = int(target_slot[b].item())
            if slot < 0 or hover_mask[b] or yaw_align_mask[b]:
                continue

            target_xy = target_rpos[b, :2]
            dist_xy = torch.norm(target_xy).item()
            if dist_xy < 1.0e-4:
                continue

            dir_to_target = target_xy / max(dist_xy, 1.0e-4)
            radial_speed = torch.dot(v_world[b, :2], dir_to_target).item()
            if radial_speed <= 0.0:
                continue

            wp_near_obstacle = False
            if 0 <= slot < parsed.waypoint_rpos.shape[1] and parsed.obstacle_active[b].any():
                active_obs = parsed.obstacle_active[b]
                wp_xy = parsed.waypoint_rpos[b, slot, :2]
                clearance = torch.norm(
                    parsed.obstacle_rpos[b, active_obs, :2] - wp_xy.unsqueeze(0),
                    dim=-1,
                ) - parsed.obstacle_radius[b, active_obs]
                wp_near_obstacle = torch.min(clearance).item() < near_obs_clearance

            allowed_speed = near_obs_speed if wp_near_obstacle else safe_speed
            brake_dist = max(min_brake_dist, radial_speed * radial_speed / (2.0 * decel) + buffer_dist)
            if dist_xy > brake_dist or radial_speed <= allowed_speed:
                continue

            excess_speed = radial_speed - allowed_speed
            strength = min(max_accel, gain * excess_speed + 0.35 * max(brake_dist - dist_xy, 0.0))
            adjusted[b, :2] = adjusted[b, :2] - dir_to_target * strength
            brake_mask[b] = True

        return adjusted, brake_mask

    # -----------------------------------------------------------------------
    # Sparse-obstacle (0/1/2) Binary Tide last-resort safety
    # -----------------------------------------------------------------------

    def _apply_binary_tide(
        self,
        parsed,
        action: torch.Tensor,
        rotation_matrix: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        angular_velocity: torch.Tensor,
        v_world: torch.Tensor,
        hover_mask: torch.Tensor,
        yaw_align_mask: torch.Tensor,
    ):
        """Binary Tide hard safety override.

        Detects when the drone is within BINARY_TIDE_MARGIN of any active
        obstacle's surface. When triggered, completely replaces the action with
        an evasive maneuver: push away laterally, climb, and brake.

        Returns:
            action:     modified action [batch, 4]
            tide_mask:  [batch] bool -- True where tide is active
        """
        tide_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        if not getattr(Config, "BINARY_TIDE_ENABLE", False):
            return action, tide_mask

        margin = float(getattr(Config, "BINARY_TIDE_MARGIN", 0.15))
        lateral_gain = float(getattr(Config, "BINARY_TIDE_LATERAL_GAIN", 2.0))
        climb_accel = float(getattr(Config, "BINARY_TIDE_CLIMB_ACCEL", 3.0))
        brake_gain = float(getattr(Config, "BINARY_TIDE_BRAKE_GAIN", 0.5))
        hold_frames = max(int(getattr(Config, "BINARY_TIDE_HOLD_FRAMES", 6)), 1)
        max_roll_rate = float(getattr(Config, "BINARY_TIDE_MAX_ROLL_RATE", 0.95))
        max_pitch_rate = float(getattr(Config, "BINARY_TIDE_MAX_PITCH_RATE", 0.95))
        thrust_boost = float(getattr(Config, "BINARY_TIDE_THRUST_BOOST", 0.08))
        g = float(getattr(Config, "MPC_GRAVITY", 9.81))
        kp_roll = float(Config.MPC_ATT_KP_ROLL)
        kp_pitch = float(Config.MPC_ATT_KP_PITCH)
        kd_roll = float(getattr(Config, "MPC_ATT_KD_ROLL", 0.0))
        kd_pitch = float(getattr(Config, "MPC_ATT_KD_PITCH", 0.0))
        action_roll_sign = float(getattr(Config, "ACTION_ROLL_SIGN", 1.0))
        action_pitch_sign = float(getattr(Config, "ACTION_PITCH_SIGN", 1.0))
        max_tilt_rad = float(getattr(Config, "BINARY_TIDE_MAX_TILT_RAD", Config.MPC_MAX_ROLL_RAD))

        # --- detect closest obstacle per environment ---
        min_clearance = torch.full((parsed.batch_size,), float("inf"), device=parsed.device)
        threat_idx = torch.full((parsed.batch_size,), -1, dtype=torch.long, device=parsed.device)
        # threat_idx: >=0 = obstacle index, -2 = X-wall, -3 = Y-wall, -1 = none
        tide_boundary_enable = getattr(Config, "BINARY_TIDE_BOUNDARY_ENABLE", True)
        boundary_gain = float(getattr(Config, "BINARY_TIDE_BOUNDARY_GAIN", 2.5))

        for b in range(parsed.batch_size):
            if hover_mask[b] or yaw_align_mask[b]:
                continue
            for i in parsed.obstacle_active[b].nonzero(as_tuple=False).view(-1):
                idx = int(i.item())
                obs_xy = parsed.obstacle_rpos[b, idx, :2]
                dist_xy = torch.norm(obs_xy).item()
                radius = float(parsed.obstacle_radius[b, idx].item())
                clearance = dist_xy - radius
                if clearance < float(min_clearance[b].item()):
                    min_clearance[b] = clearance
                    threat_idx[b] = idx

            # --- Boundary threat detection ---
            if tide_boundary_enable:
                drone_abs = self._drone_xy_from_start(parsed)[b]
                for ax, (ax_min, ax_max, ax_label) in enumerate([
                    (float(Config.ARENA_X_MIN), float(Config.ARENA_X_MAX), -2),
                    (float(Config.ARENA_Y_MIN), float(Config.ARENA_Y_MAX), -3),
                ]):
                    d_min = float(drone_abs[ax].item()) - ax_min
                    d_max = ax_max - float(drone_abs[ax].item())
                    for d in (d_min, d_max):
                        if d < margin and d < float(min_clearance[b].item()):
                            min_clearance[b] = float(d)
                            threat_idx[b] = ax_label

        # --- fresh trigger ---
        fresh_trigger = (
            (min_clearance < margin)
            & (threat_idx != -1)
            & (~hover_mask)
            & (~yaw_align_mask)
        )

        # --- hysteresis hold ---
        was_active = self._tide_hold_counter > 0
        hold_active = was_active & (self._tide_hold_counter < hold_frames)
        tide_mask = fresh_trigger | hold_active
        self._tide_hold_counter[tide_mask] += 1
        self._tide_hold_counter[~tide_mask] = 0

        if not tide_mask.any():
            return action, tide_mask

        # --- override actions for triggered environments ---
        action_modified = action.clone()
        for b in tide_mask.nonzero(as_tuple=False).view(-1):
            b_idx = int(b.item())
            t_idx = int(threat_idx[b_idx].item())
            if t_idx < -3:
                # hysteresis hold but original threat disappeared
                self._tide_hold_counter[b_idx] = 0
                tide_mask[b_idx] = False
                continue

            # Compute away direction: from obstacle center or from wall.
            if t_idx >= 0:
                # Obstacle threat
                obs_vec = parsed.obstacle_rpos[b_idx, t_idx]  # [3]
                obs_xy = obs_vec[:2]
                dist_xy = torch.norm(obs_xy).clamp_min(1.0e-6)
                away_dir = -obs_xy / dist_xy  # unit vector away from obstacle
            elif t_idx == -2:
                # X-wall boundary threat
                drone_abs = self._drone_xy_from_start(parsed)[b_idx]
                x_min = float(Config.ARENA_X_MIN)
                x_max = float(Config.ARENA_X_MAX)
                d_min = float(drone_abs[0].item()) - x_min
                d_max = x_max - float(drone_abs[0].item())
                away_dir = torch.tensor(
                    [1.0 if d_min < d_max else -1.0, 0.0],
                    device=parsed.device, dtype=torch.float32,
                )
            elif t_idx == -3:
                # Y-wall boundary threat
                drone_abs = self._drone_xy_from_start(parsed)[b_idx]
                y_min = float(Config.ARENA_Y_MIN)
                y_max = float(Config.ARENA_Y_MAX)
                d_min = float(drone_abs[1].item()) - y_min
                d_max = y_max - float(drone_abs[1].item())
                away_dir = torch.tensor(
                    [0.0, 1.0 if d_min < d_max else -1.0],
                    device=parsed.device, dtype=torch.float32,
                )
            else:
                self._tide_hold_counter[b_idx] = 0
                tide_mask[b_idx] = False
                continue

            # build world-frame acceleration
            accel_world = torch.zeros(3, device=parsed.device, dtype=torch.float32)
            push_gain = boundary_gain if t_idx < 0 else lateral_gain
            accel_world[:2] = away_dir * push_gain * g

            # braking: if drone has velocity toward obstacle, add extra push
            vel_toward = torch.dot(v_world[b_idx, :2], away_dir)  # positive = moving away
            if vel_toward < 0.0:
                accel_world[:2] += away_dir * (-vel_toward) * brake_gain

            # climb
            accel_world[2] = climb_accel

            # convert to body frame
            rot = rotation_matrix[b_idx]  # [3,3] body->world
            accel_body = rot.transpose(0, 1) @ accel_world

            # desired roll/pitch angles from body acceleration
            tide_roll_cmd = torch.clamp(accel_body[1] / g, -max_tilt_rad, max_tilt_rad)
            tide_pitch_cmd = torch.clamp(-accel_body[0] / g, -max_tilt_rad, max_tilt_rad)

            # PD rate control (reusing MPC attitude gains)
            tide_roll_rate = (
                kp_roll * (tide_roll_cmd - roll[b_idx])
                - kd_roll * angular_velocity[b_idx, 0]
            )
            tide_pitch_rate = (
                kp_pitch * (tide_pitch_cmd - pitch[b_idx])
                - kd_pitch * angular_velocity[b_idx, 1]
            )
            tide_roll_rate = torch.clamp(tide_roll_rate, -max_roll_rate, max_roll_rate)
            tide_pitch_rate = torch.clamp(tide_pitch_rate, -max_pitch_rate, max_pitch_rate)

            # set action
            action_modified[b_idx, 0] = tide_roll_rate * action_roll_sign
            action_modified[b_idx, 1] = tide_pitch_rate * action_pitch_sign
            action_modified[b_idx, 2] = 0.0  # zero yaw
            action_modified[b_idx, 3] = float(Config.HOVER_BASE_THRUST) + thrust_boost

        return action_modified, tide_mask

    def compute_action(self, obs) -> torch.Tensor:
        obs, original_shape = self._prepare_obs(obs)
        parsed = self.parser.parse(obs)
        self._ensure_state(parsed.batch_size, parsed.device)

        # ---- Obstacle-count-based strategy selection ----
        self._update_obs_strategy(parsed)
        obs_count = self._obs_count

        rotation = parsed.rotation_matrix
        v_world = torch.bmm(rotation, parsed.linear_velocity.unsqueeze(-1)).squeeze(-1)
        roll, pitch, yaw = self._extract_attitude(rotation)

        raw_target_rpos, target_slot, hover_mask = self._select_local_target(parsed)
        raw_target_rpos, target_slot, previous_dangerous_wp_mask, danger_hold_mask = (
            self._hold_recent_dangerous_waypoint(
                parsed, raw_target_rpos, target_slot, hover_mask, v_world,
            )
        )
        dangerous_wp_mask = self._build_dangerous_waypoint_mask(parsed, target_slot)
        dangerous_segment_mask = self._build_dangerous_segment_mask(parsed, raw_target_rpos, target_slot)
        near_goal_mask = self._build_goal_hover_mask(parsed, hover_mask, v_world)
        changed_target = target_slot != self._last_target_slot
        waypoint_to_waypoint = changed_target & (self._last_target_slot >= 0) & (target_slot >= 0)

        # ---- Post-danger guard (dense / 456 paths) ----
        post_danger_guard_mask = torch.zeros_like(hover_mask, dtype=torch.bool)
        if obs_count >= 3:
            if obs_count >= 4:
                guard_frames = max(self._strategy.post_danger_guard_frames, 0)
            else:
                guard_frames = max(int(getattr(Config, "MPC_POST_DANGER_SEGMENT_GUARD_FRAMES", 45)), 0)
            post_danger_transition = (
                changed_target & previous_dangerous_wp_mask & dangerous_segment_mask
                & (~hover_mask) & (target_slot >= -1)
            )
            if guard_frames > 0:
                self._post_danger_guard_counter[post_danger_transition] = guard_frames
            post_danger_guard_mask = (
                (self._post_danger_guard_counter > 0) & dangerous_segment_mask
                & (~hover_mask) & (target_slot >= -1)
            )

        hard_reset_target = changed_target & (
            (~waypoint_to_waypoint) | dangerous_wp_mask | previous_dangerous_wp_mask
        )
        self._prev_u[hard_reset_target] = 0.0
        self._wp_hold_counter[changed_target] = 0
        target_rpos, xy_scale, z_scale, high_z_mask = self._apply_startup_guard(raw_target_rpos, hover_mask)
        low_alt_mask, strong_takeoff_mask, current_up, xy_accel_limit, tilt_limit, takeoff_progress = (
            self._build_takeoff_profile(parsed, hover_mask)
        )
        yaw_align_mask, yaw_error = self._build_yaw_align_mask(yaw, hover_mask)
        target_rpos, xy_accel_limit, tilt_limit = self._apply_yaw_align_profile(
            target_rpos, current_up, xy_accel_limit, tilt_limit, yaw_align_mask,
        )
        target_rpos, avoid_mask, avoid_slot, avoid_side = self._build_obstacle_avoidance_target(
            parsed, target_rpos, hover_mask, yaw_align_mask, v_world,
        )
        ref_vel, danger_wp_brake_mask = self._build_reference_velocity(
            parsed, target_rpos, target_slot, hover_mask, yaw_align_mask,
            avoid_mask, near_goal_mask, dangerous_wp_mask, dangerous_segment_mask,
            post_danger_guard_mask=post_danger_guard_mask,
        )

        # ---- MPC solve (shared by all paths) ----
        accel_cmd = torch.zeros(parsed.batch_size, 3, device=parsed.device, dtype=torch.float32)
        for b in range(parsed.batch_size):
            xy_controller = self._mpc_xy_hover if hover_mask[b] else self._mpc_xy
            z_controller = self._mpc_z_hover if hover_mask[b] else self._mpc_z

            x_result = self._solve_axis(
                controller=xy_controller,
                position_error=float(target_rpos[b, 0].item()),
                velocity_world=float(v_world[b, 0].item()),
                prev_u=float(self._prev_u[b, 0].item()),
                ref_vel=float(ref_vel[b, 0].item()),
            )
            y_result = self._solve_axis(
                controller=xy_controller,
                position_error=float(target_rpos[b, 1].item()),
                velocity_world=float(v_world[b, 1].item()),
                prev_u=float(self._prev_u[b, 1].item()),
                ref_vel=float(ref_vel[b, 1].item()),
            )
            z_result = self._solve_axis(
                controller=z_controller,
                position_error=float(target_rpos[b, 2].item()),
                velocity_world=float(v_world[b, 2].item()),
                prev_u=float(self._prev_u[b, 2].item()),
                ref_vel=float(ref_vel[b, 2].item()),
            )

            accel_cmd[b, 0] = x_result.command.to(parsed.device)
            accel_cmd[b, 1] = y_result.command.to(parsed.device)
            accel_cmd[b, 2] = z_result.command.to(parsed.device)

        # ---- Dense / 456: danger-WP brake accel scaling ----
        if obs_count >= 3:
            brake_accel_scale = float(getattr(Config, "MPC_DANGER_WP_BRAKE_ACCEL_SCALE", 1.0))
            if brake_accel_scale != 1.0 and danger_wp_brake_mask.any():
                accel_cmd[danger_wp_brake_mask, :2] = accel_cmd[danger_wp_brake_mask, :2] * brake_accel_scale

        # ---- Dense / 456: speed-dependent target brake ----
        speed_brake_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        if obs_count >= 3:
            accel_cmd, speed_brake_mask = self._apply_speed_dependent_target_brake(
                parsed, accel_cmd, target_rpos, target_slot,
                hover_mask, yaw_align_mask, v_world,
            )

        accel_cmd, emergency_mask, tangential_mask, wall_mask = self._apply_obstacle_safety_accel(
            parsed, accel_cmd, hover_mask, yaw_align_mask, v_world,
        )
        accel_cmd = self._limit_xy_accel(accel_cmd, xy_accel_limit)
        if yaw_align_mask.any():
            align_alt = float(getattr(Config, "MPC_YAW_ALIGN_ALT", 0.45))
            min_up_accel = float(getattr(Config, "MPC_YAW_ALIGN_MIN_UP_ACCEL", 0.32))
            below_align_alt = yaw_align_mask & (current_up < align_alt)
            accel_cmd[below_align_alt, 2] = torch.maximum(
                accel_cmd[below_align_alt, 2],
                torch.full_like(accel_cmd[below_align_alt, 2], min_up_accel),
            )
        accel_cmd = self._apply_accel_slew_limit(accel_cmd, takeoff_progress)

        # ---- Dense / 456: hard obstacle safety (replaces Binary Tide for 3+ obstacles) ----
        hard_safety_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        if obs_count >= 3:
            accel_cmd, hard_safety_mask = self._apply_hard_obstacle_safety(
                parsed, accel_cmd, hover_mask, yaw_align_mask, v_world,
            )

        action, roll_cmd, pitch_cmd, accel_cmd_body = self._map_accel_to_action(
            accel_cmd, rotation, roll, pitch, parsed.angular_velocity,
            v_world, hover_mask, near_goal_mask, low_alt_mask, tilt_limit,
        )
        action[:, 3] = action[:, 3] + getattr(Config, "MPC_HOVER_THRUST_BIAS", 0.0)
        goal_hover_bias = getattr(Config, "MPC_GOAL_HOVER_THRUST_BIAS", 0.0)
        if goal_hover_bias != 0.0:
            goal_hover_like_mask = hover_mask | near_goal_mask
            action[goal_hover_like_mask, 3] = action[goal_hover_like_mask, 3] + goal_hover_bias
        if yaw_align_mask.any():
            yaw_rate = yaw_error * float(getattr(Config, "MPC_YAW_ALIGN_RATE_KP", 1.25))
            yaw_rate = torch.clamp(
                yaw_rate,
                -float(getattr(Config, "MPC_YAW_ALIGN_RATE_MAX", 0.32)),
                float(getattr(Config, "MPC_YAW_ALIGN_RATE_MAX", 0.32)),
            )
            action[yaw_align_mask, 2] = yaw_rate[yaw_align_mask] * getattr(Config, "ACTION_YAW_SIGN", 1.0)
        startup_bias = getattr(Config, "MPC_STARTUP_THRUST_BIAS", 0.0)
        startup_frames = max(int(getattr(Config, "MPC_STARTUP_FRAMES", 1)), 1)
        startup_weight = torch.clamp(1.0 - self._startup_counter.float() / float(startup_frames), 0.0, 1.0)
        action[:, 3] = action[:, 3] + startup_bias * startup_weight
        action[:, 3] = torch.clamp(action[:, 3], Config.MPC_THRUST_MIN, Config.MPC_THRUST_MAX)

        # ---- Sparse path only (0-2): Binary Tide last-resort action replacement ----
        tide_mask = torch.zeros(parsed.batch_size, dtype=torch.bool, device=parsed.device)
        if obs_count <= 2:
            action, tide_mask = self._apply_binary_tide(
                parsed, action, rotation, roll, pitch, parsed.angular_velocity,
                v_world, hover_mask, yaw_align_mask,
            )
        action[:, 3] = torch.clamp(action[:, 3], Config.MPC_THRUST_MIN, Config.MPC_THRUST_MAX)

        if tide_mask.any() and (self._log_counter <= 10 or self._log_counter % 25 == 0):
            for b in tide_mask.nonzero(as_tuple=False).view(-1)[:3]:
                b_idx = int(b.item())
                tide_msg = (
                    f"[BINARY TIDE] env={b_idx} "
                    f"act=({action[b_idx,0]:+.2f},{action[b_idx,1]:+.2f},{action[b_idx,2]:+.2f},{action[b_idx,3]:+.3f})"
                )
                if self.logger is not None:
                    self.logger.warning(tide_msg)
                print(tide_msg, flush=True)

        self._prev_u.copy_(accel_cmd)
        self._last_target_slot.copy_(target_slot)
        self._update_target_lock(parsed, target_rpos, target_slot, hover_mask)
        self._startup_counter += 1

        # Dense / 456: post-danger guard counter decrement
        if obs_count >= 3:
            self._post_danger_guard_counter = torch.clamp(self._post_danger_guard_counter - 1, min=0)

        self._log_counter += 1
        if self._log_counter <= 10 or self._log_counter % 25 == 0:
            b0 = 0
            phase_name = "HOVER" if hover_mask[b0] else "NAV"
            is_dense = obs_count >= 3
            msg = (
                f"[RuleMPC {phase_name} f{self._log_counter}] "
                f"obs_cnt={obs_count} "
                f"slot={int(target_slot[b0].item())} "
                f"lock={int(self._locked_target_slot[b0].item())} "
                f"r_ref=({raw_target_rpos[b0,0]:+.2f},{raw_target_rpos[b0,1]:+.2f},{raw_target_rpos[b0,2]:+.2f}) "
                f"r_mpc=({target_rpos[b0,0]:+.2f},{target_rpos[b0,1]:+.2f},{target_rpos[b0,2]:+.2f}) "
                f"scale=({xy_scale[b0]:.2f},{z_scale[b0]:.2f}) "
                f"zgate={int(high_z_mask[b0].item())} "
                f"up={current_up[b0]:+.2f} "
                f"lowalt={int(low_alt_mask[b0].item())} "
                f"yalign={int(yaw_align_mask[b0].item())} "
                f"danger_wp={int(dangerous_wp_mask[b0].item())}"
                + (f"/{int(danger_wp_brake_mask[b0].item())}" if is_dense else f"/{int(danger_hold_mask[b0].item())}") +
                f" danger_seg={int(dangerous_segment_mask[b0].item())} "
                + (f"pdguard={int(post_danger_guard_mask[b0].item())}/{int(self._post_danger_guard_counter[b0].item())} " if is_dense else "") +
                f"yaw={yaw[b0]:+.2f} "
                f"avoid={int(avoid_mask[b0].item())}:{int(avoid_slot[b0].item())}/{float(avoid_side[b0].item()):+.0f} "
                + ("" if is_dense else f"bld={float(self._avoid_blend_weight[b0].item()):.2f} "
                                         f"tan={int(tangential_mask[b0].item())} "
                                         f"tlock={int(self._tan_direction_lock[b0].item()):+.0f} "
                                         f"wall={int(wall_mask[b0].item())} "
                                         f"trap={int(int(emergency_mask[b0].item()) and int(wall_mask[b0].item()))} ") +
                f"emg={int(emergency_mask[b0].item())}"
                + (f"/{int(hard_safety_mask[b0].item())}" if is_dense else "") +
                f" tide={int(tide_mask[b0].item())}"
                + (f" spd_brake={int(speed_brake_mask[b0].item())}" if is_dense else "") +
                f" vref=({ref_vel[b0,0]:+.2f},{ref_vel[b0,1]:+.2f},{ref_vel[b0,2]:+.2f}) "
                f"tprog={takeoff_progress[b0]:.2f} "
                f"gnear={int(near_goal_mask[b0].item())} "
                f"v=({v_world[b0,0]:+.2f},{v_world[b0,1]:+.2f},{v_world[b0,2]:+.2f}) "
                f"a_w=({accel_cmd[b0,0]:+.2f},{accel_cmd[b0,1]:+.2f},{accel_cmd[b0,2]:+.2f}) "
                f"a_b=({accel_cmd_body[b0,0]:+.2f},{accel_cmd_body[b0,1]:+.2f},{accel_cmd_body[b0,2]:+.2f}) "
                f"att_cmd=({roll_cmd[b0]:+.2f},{pitch_cmd[b0]:+.2f}) "
                f"act=({action[b0,0]:+.2f},{action[b0,1]:+.2f},{action[b0,2]:+.2f},{action[b0,3]:+.3f})"
            )
            if self.logger is not None:
                self.logger.info(msg)
            print(msg, flush=True)

        return self._restore_shape(action, original_shape)


__all__ = ["RuleMPCController"]
