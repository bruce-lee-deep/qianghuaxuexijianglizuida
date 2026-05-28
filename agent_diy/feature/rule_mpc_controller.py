#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agent_diy.conf.conf import Config
from agent_diy.feature.observation_parser import ObservationParser


@dataclass
class MPCDebug:
    sub_target: torch.Tensor
    desired_velocity: torch.Tensor
    accel_cmd: torch.Tensor


class RuleMPCController:
    """Pure-rule waypoint selection + linear MPC + cascaded PID/P control."""

    def __init__(self, logger=None):
        self.logger = logger
        self.parser = ObservationParser()

        self._vel_int = None
        self._vel_prev_err = None
        self._hover_pos_int = None
        self._hover_pos_prev_err = None
        self._prev_v_cmd = None
        self._hold_counter = None
        self._holding_wp = None

        self.dt = float(Config.CTRL_DT)
        self.horizon = int(Config.MPC_HORIZON)
        self._setup_mpc_mats()

    def _setup_mpc_mats(self):
        dt = self.dt
        A1 = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float32)
        B1 = torch.tensor([[dt], [1.0]], dtype=torch.float32)
        C1 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        Sx = []
        Su_cols = []
        A_pow = torch.eye(2, dtype=torch.float32)
        for i in range(self.horizon):
            A_pow = A_pow @ A1
            Sx.append(C1 @ A_pow)

            row_blocks = []
            for j in range(self.horizon):
                if j <= i:
                    A_term = torch.matrix_power(A1, i - j)
                    row_blocks.append(C1 @ A_term @ B1)
                else:
                    row_blocks.append(torch.zeros(1, 1, dtype=torch.float32))
            Su_cols.append(torch.cat(row_blocks, dim=1))

        self._mpc_Sx = torch.cat(Sx, dim=0)
        self._mpc_Su = torch.cat(Su_cols, dim=0)
        self._mpc_D = torch.eye(self.horizon, dtype=torch.float32)
        self._mpc_D[1:, :-1] -= torch.eye(self.horizon - 1, dtype=torch.float32)

    def _ensure_state(self, batch_size: int, device: torch.device):
        if self._vel_int is not None and self._vel_int.shape[0] == batch_size:
            return
        self._vel_int = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
        self._vel_prev_err = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
        self._hover_pos_int = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
        self._hover_pos_prev_err = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
        self._prev_v_cmd = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
        self._hold_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._holding_wp = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def reset(self):
        self._vel_int = None
        self._vel_prev_err = None
        self._hover_pos_int = None
        self._hover_pos_prev_err = None
        self._prev_v_cmd = None
        self._hold_counter = None
        self._holding_wp = None

    def compute_action(self, obs) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        original_shape = None
        if obs.dim() == 3:
            original_shape = obs.shape
            obs = obs.view(obs.shape[0] * obs.shape[1], -1)

        parsed = self.parser.parse(obs)
        self._ensure_state(parsed.batch_size, parsed.device)

        sub_target, target_dist, target_is_wp, speed_limit = self._select_sub_target(parsed)
        hover_mask = parsed.phase_onehot[:, 1] > 0.5
        sub_target = torch.where(hover_mask.unsqueeze(-1), parsed.goal_rpos, sub_target)
        speed_limit = torch.where(
            hover_mask,
            torch.full_like(speed_limit, Config.HOVER_VEL_CMD_LIM[0]),
            speed_limit,
        )

        nav_v_cmd = self._solve_linear_mpc(sub_target, parsed.linear_velocity, speed_limit)
        hover_v_cmd = self._hover_position_pid(parsed.goal_rpos, parsed.linear_velocity)
        v_cmd = torch.where(hover_mask.unsqueeze(-1), hover_v_cmd, nav_v_cmd)

        xy_speed = torch.norm(parsed.linear_velocity[:, :2], dim=-1)
        hold_trigger = target_is_wp & (
            (target_dist < Config.WAYPOINT_HOLD_DIST)
            | ((target_dist < Config.WAYPOINT_SWITCH_RADIUS) & (xy_speed < Config.WAYPOINT_HOLD_SPEED))
        )
        self._hold_counter[hold_trigger] += 1
        self._hold_counter[~hold_trigger] = 0
        self._holding_wp = self._hold_counter >= Config.WAYPOINT_HOLD_FRAMES
        hold_mask = self._holding_wp & (~hover_mask)
        if hold_mask.any():
            hold_v_cmd = self._hover_position_pid(sub_target, parsed.linear_velocity)
            v_cmd = torch.where(hold_mask.unsqueeze(-1), hold_v_cmd, v_cmd)
        self._holding_wp = hold_mask

        accel_cmd = self._velocity_pid(v_cmd, parsed.linear_velocity, hover_mask)
        action = self._map_accel_to_action(accel_cmd, parsed.current_rpy)
        action = torch.clamp(action, -1.0, 1.0)

        self._prev_v_cmd = v_cmd.detach().clone()
        if original_shape is not None:
            env_num, agent_num = original_shape[0], original_shape[1]
            action = action.view(env_num, agent_num, 4)
        return action

    def _speed_cap_by_distance(self, dist: torch.Tensor) -> torch.Tensor:
        cap = torch.full_like(dist, Config.MPC_SPEED_FAR)
        cap = torch.where(dist <= 1.2, torch.full_like(cap, Config.MPC_SPEED_MID), cap)
        cap = torch.where(dist <= 0.8, torch.full_like(cap, Config.MPC_SPEED_NEAR), cap)
        cap = torch.where(dist <= Config.SHORT_DIST_STRONG_BRAKE, torch.full_like(cap, Config.MPC_SPEED_CLOSE), cap)
        return cap

    def _select_sub_target(self, parsed):
        wp_rpos = parsed.waypoint_rpos
        wp_pending = parsed.waypoint_active & (~parsed.waypoint_visited)
        wp_dist = torch.norm(wp_rpos, dim=-1)
        large = torch.full_like(wp_dist, 1.0e6)
        candidate_dist = torch.where(wp_pending, wp_dist, large)
        nearest_idx = torch.argmin(candidate_dist, dim=-1)

        gather_idx = nearest_idx.view(-1, 1, 1).expand(-1, 1, 3)
        nearest_wp = torch.gather(wp_rpos, 1, gather_idx).squeeze(1)
        has_pending = wp_pending.any(dim=-1, keepdim=True)
        raw_target = torch.where(has_pending, nearest_wp, parsed.goal_rpos)
        target_is_wp = has_pending.squeeze(-1)

        dist = torch.norm(raw_target[:, :2], dim=-1)
        lookahead_far = torch.full_like(dist, Config.LOOKAHEAD_DIST_FAR)
        lookahead_near = torch.full_like(dist, Config.LOOKAHEAD_DIST_NEAR)
        lookahead = torch.where(dist > 0.8, lookahead_far, lookahead_near)

        dir_xy = raw_target[:, :2] / torch.clamp(dist.unsqueeze(-1), min=1.0e-6)
        lookahead_xy = dir_xy * torch.minimum(dist, lookahead).unsqueeze(-1)
        sub_target = raw_target.clone()
        sub_target[:, :2] = torch.where(
            target_is_wp.unsqueeze(-1),
            lookahead_xy,
            raw_target[:, :2],
        )

        speed_limit = self._speed_cap_by_distance(dist)
        goal_xy = torch.norm(parsed.goal_rpos[:, :2], dim=-1)
        goal_dir = parsed.goal_rpos[:, :2] / torch.clamp(goal_xy.unsqueeze(-1), min=1.0e-6)
        cos_turn = torch.sum(dir_xy * goal_dir, dim=-1)
        turn_scale = torch.where(
            cos_turn < Config.TURN_SLOWDOWN_COS,
            torch.full_like(cos_turn, Config.TURN_SPEED_SCALE),
            torch.ones_like(cos_turn),
        )
        speed_limit = torch.where(target_is_wp, speed_limit * turn_scale, speed_limit)
        return sub_target, dist, target_is_wp, speed_limit

    def _solve_linear_mpc(self, sub_target: torch.Tensor, current_vel: torch.Tensor, speed_limit: torch.Tensor) -> torch.Tensor:
        device = sub_target.device
        batch = sub_target.shape[0]
        Sx = self._mpc_Sx.to(device)
        Su = self._mpc_Su.to(device)
        D = self._mpc_D.to(device)

        pos_w = torch.tensor(Config.MPC_POS_WEIGHT, dtype=torch.float32, device=device)
        vel_w = torch.tensor(Config.MPC_VEL_WEIGHT, dtype=torch.float32, device=device)
        smooth_w = torch.tensor(Config.MPC_SMOOTH_WEIGHT, dtype=torch.float32, device=device)
        term_pos_w = torch.tensor(Config.MPC_TERMINAL_POS_WEIGHT, dtype=torch.float32, device=device)
        term_vel_w = torch.tensor(Config.MPC_TERMINAL_VEL_WEIGHT, dtype=torch.float32, device=device)

        v_lim = torch.stack(
            [
                speed_limit.clamp(max=Config.MPC_VEL_LIMIT_XY),
                speed_limit.clamp(max=Config.MPC_VEL_LIMIT_XY),
                torch.full_like(speed_limit, Config.MPC_VEL_LIMIT_Z),
            ],
            dim=-1,
        )

        axes_cmd = []
        for axis in range(3):
            x0 = torch.stack([torch.zeros(batch, device=device), current_vel[:, axis]], dim=-1)
            x_ref = torch.stack([sub_target[:, axis], torch.zeros(batch, device=device)], dim=-1)

            q = torch.ones(self.horizon, dtype=torch.float32, device=device) * pos_w[axis]
            q[-1] = term_pos_w[axis]
            r = torch.ones(self.horizon, dtype=torch.float32, device=device) * smooth_w[axis]

            Q = torch.diag(q)
            vel_ref_penalty = vel_w[axis] * torch.eye(self.horizon, dtype=torch.float32, device=device)
            R = torch.diag(r) + vel_ref_penalty
            H = Su.T @ Q @ Su + D.T @ R @ D + 1.0e-4 * torch.eye(self.horizon, device=device)

            pos_free = (Sx @ x0.T).T
            pos_ref = x_ref[:, 0:1].repeat(1, self.horizon)
            g = torch.matmul((pos_free - pos_ref), Q @ Su).unsqueeze(-1)

            u_seq = []
            H_inv = torch.linalg.inv(H)
            for b in range(batch):
                rhs = -g[b].squeeze(-1)
                u_star = H_inv @ rhs
                u_prev = self._prev_v_cmd[b, axis]
                u_star = torch.clamp(u_star, -v_lim[b, axis], v_lim[b, axis])
                u_star[0] = torch.clamp(
                    0.75 * u_star[0] + 0.25 * u_prev,
                    -v_lim[b, axis],
                    v_lim[b, axis],
                )
                u_seq.append(u_star[0])
            axes_cmd.append(torch.stack(u_seq, dim=0))

        return torch.stack(axes_cmd, dim=-1)

    def _velocity_pid(self, v_cmd: torch.Tensor, current_vel: torch.Tensor, hover_mask: torch.Tensor) -> torch.Tensor:
        err = v_cmd - current_vel

        kp = torch.tensor(Config.VEL_KP, dtype=torch.float32, device=v_cmd.device)
        ki = torch.tensor(Config.VEL_KI, dtype=torch.float32, device=v_cmd.device)
        kd = torch.tensor(Config.VEL_KD, dtype=torch.float32, device=v_cmd.device)
        int_lim = torch.tensor(Config.VEL_INT_LIM, dtype=torch.float32, device=v_cmd.device)
        acc_lim = torch.tensor(Config.ACC_CMD_LIM, dtype=torch.float32, device=v_cmd.device)

        self._vel_int = self._vel_int + err * self.dt
        self._vel_int = torch.clamp(self._vel_int, -int_lim, int_lim)
        derr = (err - self._vel_prev_err) / self.dt
        self._vel_prev_err = err.detach().clone()

        accel_cmd = kp * err + ki * self._vel_int + kd * derr
        accel_cmd = torch.clamp(accel_cmd, -acc_lim, acc_lim)

        # final hover: slightly stronger damping
        accel_cmd[hover_mask] = 1.15 * accel_cmd[hover_mask]
        return accel_cmd

    def _hover_position_pid(self, goal_rpos: torch.Tensor, current_vel: torch.Tensor) -> torch.Tensor:
        pos_err = goal_rpos
        kp = torch.tensor(Config.HOVER_POS_KP, dtype=torch.float32, device=goal_rpos.device)
        ki = torch.tensor(Config.HOVER_POS_KI, dtype=torch.float32, device=goal_rpos.device)
        kd = torch.tensor(Config.HOVER_POS_KD, dtype=torch.float32, device=goal_rpos.device)
        int_lim = torch.tensor(Config.HOVER_POS_INT_LIM, dtype=torch.float32, device=goal_rpos.device)
        vel_lim = torch.tensor(Config.HOVER_VEL_CMD_LIM, dtype=torch.float32, device=goal_rpos.device)

        self._hover_pos_int = self._hover_pos_int + pos_err * self.dt
        self._hover_pos_int = torch.clamp(self._hover_pos_int, -int_lim, int_lim)
        derr = (pos_err - self._hover_pos_prev_err) / self.dt
        self._hover_pos_prev_err = pos_err.detach().clone()

        v_cmd = kp * pos_err + ki * self._hover_pos_int + kd * derr - 0.6 * current_vel
        return torch.clamp(v_cmd, -vel_lim, vel_lim)

    def _map_accel_to_action(self, accel_cmd: torch.Tensor, current_rpy: torch.Tensor) -> torch.Tensor:
        ax = accel_cmd[:, 0]
        ay = accel_cmd[:, 1]
        az = accel_cmd[:, 2]

        pitch_cmd = torch.clamp(ax / Config.GRAVITY, -Config.MAX_PITCH_CMD, Config.MAX_PITCH_CMD)
        roll_cmd = torch.clamp(-ay / Config.GRAVITY, -Config.MAX_ROLL_CMD, Config.MAX_ROLL_CMD)

        roll_rate = Config.ROLL_P_RATE_KP * (roll_cmd - current_rpy[:, 0])
        pitch_rate = Config.PITCH_P_RATE_KP * (pitch_cmd - current_rpy[:, 1])
        roll_rate = torch.clamp(roll_rate, -Config.MAX_ROLL_RATE_CMD, Config.MAX_ROLL_RATE_CMD)
        pitch_rate = torch.clamp(pitch_rate, -Config.MAX_PITCH_RATE_CMD, Config.MAX_PITCH_RATE_CMD)

        thrust = Config.HOVER_THRUST_BIAS + Config.THRUST_SCALE * az
        thrust = torch.clamp(thrust, Config.THRUST_MIN, Config.THRUST_MAX)

        yaw_rate = torch.zeros_like(roll_rate)
        return torch.stack([roll_rate, pitch_rate, yaw_rate, thrust], dim=-1)


__all__ = ["RuleMPCController", "MPCDebug"]
