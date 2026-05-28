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
        self._log_counter = 0

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

    def reset(self, batch_size: int = None, device: torch.device = None):
        if batch_size is not None and device is not None:
            self._prev_u = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
            self._last_target_slot = torch.full((batch_size,), -2, dtype=torch.long, device=device)
            self._startup_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._yaw_align_hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._locked_target_slot = torch.full((batch_size,), -2, dtype=torch.long, device=device)
            self._wp_hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            self._prev_u = None
            self._last_target_slot = None
            self._startup_counter = None
            self._yaw_align_hold_counter = None
            self._locked_target_slot = None
            self._wp_hold_counter = None

    def _select_local_target(self, parsed):
        batch_size = parsed.batch_size
        device = parsed.device

        target_rpos = parsed.goal_rpos.clone()
        target_slot = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        hover_mask = parsed.phase_hover > 0.5
        target_rpos[hover_mask] = 0.0
        target_slot[hover_mask] = -99

        pending_mask = parsed.waypoint_active & (~parsed.waypoint_visited)
        if getattr(Config, "DIRECT_GOAL_MODE", False):
            return target_rpos, target_slot, hover_mask

        for b in range(batch_size):
            if hover_mask[b]:
                self._locked_target_slot[b] = -99
                continue
            pending_idx = pending_mask[b].nonzero(as_tuple=False).view(-1)
            if pending_idx.numel() == 0:
                self._locked_target_slot[b] = -1
                continue

            pending_points = parsed.waypoint_rpos[b, pending_idx]
            distances = torch.norm(pending_points, dim=-1)
            best_local = torch.argmin(distances)
            best_idx = int(pending_idx[best_local].item())
            target_rpos[b] = parsed.waypoint_rpos[b, best_idx]
            target_slot[b] = best_idx
            self._locked_target_slot[b] = best_idx

        return target_rpos, target_slot, hover_mask

    def _update_target_lock(self, parsed, target_rpos: torch.Tensor, target_slot: torch.Tensor, hover_mask: torch.Tensor):
        del parsed, target_rpos, hover_mask
        self._locked_target_slot.copy_(target_slot)
        self._wp_hold_counter.zero_()

    def _solve_axis(self, controller: LinearAxisMPC, position_error: float, velocity_world: float, prev_u: float):
        x0 = np.array([position_error, velocity_world], dtype=np.float64)
        return controller.solve(x0=x0, u_prev=prev_u, ref_pos=0.0, ref_vel=0.0)

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
    def _limit_xy_accel(accel_cmd: torch.Tensor, xy_accel_limit: torch.Tensor):
        xy = accel_cmd[:, :2]
        xy_norm = torch.norm(xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
        scale = torch.clamp(xy_accel_limit.unsqueeze(-1) / xy_norm, max=1.0)
        accel_cmd[:, :2] = xy * scale
        return accel_cmd

    def _apply_accel_slew_limit(self, accel_cmd: torch.Tensor, takeoff_progress: torch.Tensor):
        xy_slew_low = float(getattr(Config, "MPC_TAKEOFF_XY_SLEW_LIMIT", 0.04))
        xy_slew_high = float(getattr(Config, "MPC_XY_SLEW_LIMIT", 0.10))
        z_slew_low = float(getattr(Config, "MPC_TAKEOFF_Z_SLEW_LIMIT", 0.08))
        z_slew_high = float(getattr(Config, "MPC_Z_SLEW_LIMIT", 0.16))

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

    def compute_action(self, obs) -> torch.Tensor:
        obs, original_shape = self._prepare_obs(obs)
        parsed = self.parser.parse(obs)
        self._ensure_state(parsed.batch_size, parsed.device)

        rotation = parsed.rotation_matrix
        v_world = torch.bmm(rotation, parsed.linear_velocity.unsqueeze(-1)).squeeze(-1)
        roll, pitch, yaw = self._extract_attitude(rotation)

        raw_target_rpos, target_slot, hover_mask = self._select_local_target(parsed)
        near_goal_mask = self._build_goal_hover_mask(parsed, hover_mask, v_world)
        changed_target = target_slot != self._last_target_slot
        self._prev_u[changed_target] = 0.0
        self._wp_hold_counter[changed_target] = 0
        target_rpos, xy_scale, z_scale, high_z_mask = self._apply_startup_guard(raw_target_rpos, hover_mask)
        low_alt_mask, strong_takeoff_mask, current_up, xy_accel_limit, tilt_limit, takeoff_progress = (
            self._build_takeoff_profile(parsed, hover_mask)
        )
        yaw_align_mask, yaw_error = self._build_yaw_align_mask(yaw, hover_mask)
        target_rpos, xy_accel_limit, tilt_limit = self._apply_yaw_align_profile(
            target_rpos,
            current_up,
            xy_accel_limit,
            tilt_limit,
            yaw_align_mask,
        )

        accel_cmd = torch.zeros(parsed.batch_size, 3, device=parsed.device, dtype=torch.float32)
        for b in range(parsed.batch_size):
            xy_controller = self._mpc_xy_hover if hover_mask[b] else self._mpc_xy
            z_controller = self._mpc_z_hover if hover_mask[b] else self._mpc_z

            x_result = self._solve_axis(
                controller=xy_controller,
                position_error=float(target_rpos[b, 0].item()),
                velocity_world=float(v_world[b, 0].item()),
                prev_u=float(self._prev_u[b, 0].item()),
            )
            y_result = self._solve_axis(
                controller=xy_controller,
                position_error=float(target_rpos[b, 1].item()),
                velocity_world=float(v_world[b, 1].item()),
                prev_u=float(self._prev_u[b, 1].item()),
            )
            z_result = self._solve_axis(
                controller=z_controller,
                position_error=float(target_rpos[b, 2].item()),
                velocity_world=float(v_world[b, 2].item()),
                prev_u=float(self._prev_u[b, 2].item()),
            )

            accel_cmd[b, 0] = x_result.command.to(parsed.device)
            accel_cmd[b, 1] = y_result.command.to(parsed.device)
            accel_cmd[b, 2] = z_result.command.to(parsed.device)

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
        action, roll_cmd, pitch_cmd, accel_cmd_body = self._map_accel_to_action(
            accel_cmd,
            rotation,
            roll,
            pitch,
            parsed.angular_velocity,
            v_world,
            hover_mask,
            near_goal_mask,
            low_alt_mask,
            tilt_limit,
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

        self._prev_u.copy_(accel_cmd)
        self._last_target_slot.copy_(target_slot)
        self._update_target_lock(parsed, target_rpos, target_slot, hover_mask)
        self._startup_counter += 1

        self._log_counter += 1
        if self._log_counter <= 10 or self._log_counter % 25 == 0:
            b0 = 0
            phase_name = "HOVER" if hover_mask[b0] else "NAV"
            msg = (
                f"[RuleMPC {phase_name} f{self._log_counter}] "
                f"slot={int(target_slot[b0].item())} "
                f"lock={int(self._locked_target_slot[b0].item())} "
                f"r_ref=({raw_target_rpos[b0,0]:+.2f},{raw_target_rpos[b0,1]:+.2f},{raw_target_rpos[b0,2]:+.2f}) "
                f"r_mpc=({target_rpos[b0,0]:+.2f},{target_rpos[b0,1]:+.2f},{target_rpos[b0,2]:+.2f}) "
                f"scale=({xy_scale[b0]:.2f},{z_scale[b0]:.2f}) "
                f"zgate={int(high_z_mask[b0].item())} "
                f"up={current_up[b0]:+.2f} "
                f"lowalt={int(low_alt_mask[b0].item())} "
                f"yalign={int(yaw_align_mask[b0].item())} "
                f"yaw={yaw[b0]:+.2f} "
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
