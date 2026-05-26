#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Top-level hierarchical controller.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from agent_ppo.conf.conf import Config
from agent_ppo.feature.observation_parser import ObservationParser
from agent_ppo.feature.potential_field_planner import PotentialFieldPlanner
from agent_ppo.feature.velocity_controller import VelocityController
from agent_ppo.feature.waypoint_planner import WaypointPlanner
from agent_ppo.feature.waypoint_sequencer import WaypointSequencer


class HierarchicalController:
    """Rule-based hierarchical drone navigation controller."""

    def __init__(self, logger=None):
        self.logger = logger
        self.parser = ObservationParser()
        self.waypoint_seq = WaypointSequencer()
        self.planner = PotentialFieldPlanner()
        self.vel_ctrl = VelocityController()
        self.wp_planner = WaypointPlanner()

        self._straight_line = Config.STRAIGHT_LINE_MODE
        self._direct_goal_mode = getattr(Config, "DIRECT_GOAL_MODE", False)
        self._hover_test = Config.HOVER_TEST
        self._param_test = Config.PARAM_TEST

        self._custom_wp_offsets = None
        self._wp_offsets = None
        self._wp_index = None
        self._wp_phase = None
        self._hover_timer = None
        self._hover_ref_target_rpos = None
        self._wp_from_fly = None

        L = self.logger.info if self.logger else print
        L(
            f"分层控制器已就绪 直线={self._straight_line} "
            f"单目标={self._direct_goal_mode} 悬停测试={self._hover_test} "
            f"roll_sign={getattr(Config, 'ACTION_ROLL_SIGN', 1.0):+.1f} "
            f"pitch_sign={getattr(Config, 'ACTION_PITCH_SIGN', 1.0):+.1f}"
        )

    def compute_action(self, obs) -> torch.Tensor:
        if self._param_test:
            return self._compute_action_param_test(obs)
        if self._hover_test:
            return self._compute_action_hover_test(obs)
        if self._straight_line:
            return self._compute_action_straight_line(obs)
        return self._compute_action_full(obs)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _extract_attitude(rotation_matrix: torch.Tensor):
        """Extract roll/pitch/yaw from body->world rotation."""
        r20 = torch.clamp(rotation_matrix[:, 2, 0], -0.999, 0.999)
        pitch = torch.asin(-r20)
        roll = torch.atan2(rotation_matrix[:, 2, 1], rotation_matrix[:, 2, 2])
        yaw = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
        return roll, pitch, yaw

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
        env_num, agent_num = original_shape[0], original_shape[1]
        return action.view(env_num, agent_num, 4)

    @staticmethod
    def _apply_action_axis_signs(action: torch.Tensor) -> torch.Tensor:
        signed_action = action.clone()
        signed_action[:, 0] = signed_action[:, 0] * getattr(Config, "ACTION_ROLL_SIGN", 1.0)
        signed_action[:, 1] = signed_action[:, 1] * getattr(Config, "ACTION_PITCH_SIGN", 1.0)
        signed_action[:, 2] = signed_action[:, 2] * getattr(Config, "ACTION_YAW_SIGN", 1.0)
        return signed_action

    @staticmethod
    def _clip_xy_vector(xy_vec: torch.Tensor, max_speed: float) -> torch.Tensor:
        if max_speed <= 1.0e-6:
            return torch.zeros_like(xy_vec)
        xy_norm = torch.norm(xy_vec, dim=-1, keepdim=True)
        scale = torch.clamp(max_speed / torch.clamp(xy_norm, min=1.0e-6), max=1.0)
        return xy_vec * scale

    def _compute_braked_desired_vxy(
        self,
        target_xy: torch.Tensor,
        current_vxy_world: torch.Tensor,
        kp_xy: float,
        max_xy_speed: float,
        brake_dist: float,
        brake_decel: float,
        reverse_gain: float,
    ):
        dist_xy = torch.norm(target_xy, dim=-1)
        dir_xy = target_xy / torch.clamp(dist_xy.unsqueeze(-1), min=1.0e-6)
        nominal_vxy = self._clip_xy_vector(kp_xy * target_xy, max_xy_speed)
        nominal_speed = torch.norm(nominal_vxy, dim=-1)

        radial_speed = torch.sum(current_vxy_world * dir_xy, dim=-1)
        effective_dist = torch.clamp(
            dist_xy - getattr(Config, "BRAKE_DEADBAND_DIST_XY", 0.05),
            min=0.0,
        )
        brake_decel = max(brake_decel, 1.0e-3)
        speed_cap = torch.sqrt(torch.clamp(2.0 * brake_decel * effective_dist, min=0.0))
        speed_cap = torch.clamp(speed_cap, max=max_xy_speed)
        stopping_dist = torch.clamp(radial_speed, min=0.0) ** 2 / (2.0 * brake_decel)

        brake_active = (dist_xy < brake_dist) | (stopping_dist > effective_dist)
        max_reverse = getattr(Config, "BRAKE_MAX_REVERSE_SPEED", max_xy_speed * 0.5)
        reverse_speed = torch.clamp(
            reverse_gain * torch.clamp(radial_speed - speed_cap, min=0.0),
            min=0.0,
            max=max_reverse,
        )
        desired_speed = torch.where(
            brake_active,
            torch.minimum(nominal_speed, speed_cap) - reverse_speed,
            nominal_speed,
        )
        desired_speed = torch.clamp(desired_speed, min=-max_reverse, max=max_xy_speed)
        desired_vxy = dir_xy * desired_speed.unsqueeze(-1)
        desired_vxy = torch.where(
            dist_xy.unsqueeze(-1) > 1.0e-6,
            desired_vxy,
            torch.zeros_like(desired_vxy),
        )
        return desired_vxy, brake_active, radial_speed, speed_cap, stopping_dist, desired_speed

    def reset_envs(self, env_mask: torch.Tensor):
        """Reset controller state for completed environments inside a batch."""
        if self._wp_offsets is None or env_mask is None:
            return

        if env_mask.dtype != torch.bool:
            env_mask = env_mask.to(dtype=torch.bool)
        if env_mask.ndim != 1 or env_mask.numel() != len(self._wp_offsets):
            raise ValueError("env_mask shape does not match controller state")

        done_indices = env_mask.nonzero(as_tuple=True)[0].detach().cpu().tolist()
        for idx in done_indices:
            self._wp_offsets[idx] = None

        self._target_idx[env_mask] = 0
        self._needs_replan[env_mask] = True
        self._wp_index[env_mask] = 0
        self._wp_phase[env_mask] = 0
        self._hover_timer[env_mask] = 0
        self._hover_ref_target_rpos[env_mask].zero_()
        self._wp_from_fly[env_mask] = False

    def _compute_action_param_test(self, obs) -> torch.Tensor:
        obs, original_shape = self._prepare_obs(obs)
        parsed = self.parser.parse(obs)
        batch_size = parsed.batch_size
        device = parsed.device

        if not hasattr(self, "_pt_phase"):
            self._pt_phase = 0
            self._pt_frame = 0
            self._pt_start_v = None

        self._pt_frame += 1
        duration = 200
        if self._pt_frame > duration:
            self._pt_frame = 1
            self._pt_phase = (self._pt_phase + 1) % 4
            self._pt_start_v = None
            phase_names = {
                0: "pitch方向测试",
                1: "roll方向测试",
                2: "推力+测试",
                3: "推力-测试",
            }
            msg = f"[参数测试] >>> 进入阶段{self._pt_phase}: {phase_names[self._pt_phase]}"
            if self.logger:
                self.logger.info(msg)
            print(msg, flush=True)

        base = Config.HOVER_BASE_THRUST
        if self._pt_phase == 0:
            roll_cmd, pitch_cmd, thrust_cmd = 0.0, -0.10, base
        elif self._pt_phase == 1:
            roll_cmd, pitch_cmd, thrust_cmd = 0.10, 0.0, base
        elif self._pt_phase == 2:
            roll_cmd, pitch_cmd, thrust_cmd = 0.0, 0.0, base + 0.05
        else:
            roll_cmd, pitch_cmd, thrust_cmd = 0.0, 0.0, base - 0.05

        action = torch.zeros(batch_size, 4, device=device)
        action[:, 0] = roll_cmd
        action[:, 1] = pitch_cmd
        action[:, 3] = thrust_cmd
        action = self._apply_action_axis_signs(action)

        vel = parsed.linear_velocity.clone()
        if self._pt_start_v is None:
            self._pt_start_v = vel
        if self._pt_frame % 40 == 1:
            dv = (vel - self._pt_start_v)[0]
            msg = (
                f"[参数测试 阶段{self._pt_phase} f{self._pt_frame}] "
                f"当前速度=({vel[0,0]:.3f},{vel[0,1]:.3f},{vel[0,2]:.3f}) "
                f"速度变化=({dv[0]:.3f},{dv[1]:.3f},{dv[2]:.3f}) "
                f"控制器指令=({roll_cmd:.2f},{pitch_cmd:.2f},th={thrust_cmd:.4f}) "
                f"下发=({action[0,0]:.2f},{action[0,1]:.2f},th={action[0,3]:.4f})"
            )
            if self.logger:
                self.logger.info(msg)
            print(msg, flush=True)

        return self._restore_shape(action, original_shape)

    def _compute_action_straight_line(self, obs) -> torch.Tensor:
        obs, original_shape = self._prepare_obs(obs)
        parsed = self.parser.parse(obs)

        target_rpos = parsed.target_rpos
        action, debug = self._compute_tracking_action(parsed, target_rpos, phase="fly")

        if not hasattr(self, "_sl_fc"):
            self._sl_fc = 0
        self._sl_fc += 1
        if self._sl_fc <= 10 or self._sl_fc % 30 == 0:
            msg = (
                f">>> 定点飞行 f{self._sl_fc} d_xy={debug['dist_xy'][0]:.2f} "
                f"tz={debug['tz'][0]:.2f} yaw_e={debug['yaw_error'][0]:.2f} "
                f"act=({action[0,0]:.2f},{action[0,1]:.2f},{action[0,2]:.2f},{action[0,3]:.2f})"
            )
            if self.logger:
                self.logger.info(msg)
            print(msg, flush=True)

        return self._restore_shape(action, original_shape)

    def _compute_action_hover_test(self, obs) -> torch.Tensor:
        obs, original_shape = self._prepare_obs(obs)
        parsed = self.parser.parse(obs)
        batch_size = parsed.batch_size
        device = parsed.device

        fixed_thrust = Config.HOVER_BASE_THRUST
        action = torch.zeros(batch_size, 4, device=device)
        action[:, 3] = fixed_thrust
        action = self._apply_action_axis_signs(action)

        if not hasattr(self, "_ht_cnt"):
            self._ht_cnt = 0
        self._ht_cnt += 1
        if self._ht_cnt <= 10 or self._ht_cnt % 30 == 0:
            vz = parsed.linear_velocity[:, 2]
            msg = (
                f"[悬停测试 f{self._ht_cnt}] "
                f"vz={vz[0].item():.3f} thrust={fixed_thrust:.3f}"
            )
            if self.logger:
                self.logger.info(msg)
            print(msg, flush=True)

        return self._restore_shape(action, original_shape)

    def _compute_tracking_action(self, parsed, target_rpos, phase: str, hover_ref_target_rpos=None):
        tx = target_rpos[:, 0]
        ty = target_rpos[:, 1]
        tz = target_rpos[:, 2]
        dist_xy = torch.norm(target_rpos[:, :2], dim=-1)

        rotation = parsed.rotation_matrix
        current_roll, current_pitch, current_yaw = self._extract_attitude(rotation)
        v_body = parsed.linear_velocity
        vx_b = v_body[:, 0]
        vy_b = v_body[:, 1]
        vz = v_body[:, 2]
        v_world = torch.bmm(rotation, v_body.unsqueeze(-1)).squeeze(-1)
        hover_xy_speed = torch.sqrt(vx_b**2 + vy_b**2)

        target_heading = torch.atan2(ty, tx)
        yaw_error = self._wrap_angle(target_heading - current_yaw)
        if phase == "fly":
            vel_heading = torch.atan2(v_world[:, 1], v_world[:, 0])
            yaw_error_fly = self._wrap_angle(vel_heading - current_yaw)
            yaw_rate = torch.clamp(-0.8 * yaw_error_fly, -0.25, 0.25)
        elif phase == "hover":
            yaw_rate = torch.clamp(-0.5 * yaw_error, -0.15, 0.15)
        else:
            yaw_rate = torch.clamp(-0.8 * yaw_error, -0.25, 0.25)

        max_a = Config.MAX_TILT_ANGLE
        rise_max_a = min(max_a, getattr(Config, "RISE_MAX_TILT_ANGLE", max_a))
        hover_max_a = min(max_a, getattr(Config, "HOVER_MAX_TILT_ANGLE", max_a))

        if phase == "fly":
            max_xy_speed = getattr(Config, "FLY_MAX_XY_SPEED", Config.MAX_XY_VEL)
            phase_kp_xy = getattr(Config, "FLY_POS_KP_XY", Config.KP_POS)
            brake_dist = getattr(Config, "FLY_BRAKE_DIST_XY", 1.0)
            brake_decel = getattr(Config, "FLY_BRAKE_DECEL", 1.5)
            brake_reverse_gain = getattr(Config, "FLY_BRAKE_REVERSE_GAIN", 0.8)
            base_max_tilt = max_a
            brake_max_tilt = max(
                base_max_tilt,
                getattr(Config, "FLY_BRAKE_MAX_TILT_ANGLE", base_max_tilt),
            )
            desired_vz = torch.clamp(
                getattr(Config, "FLY_DESIRED_VZ_GAIN", 0.15) * tz,
                -getattr(Config, "FLY_MAX_VZ", 0.3),
                getattr(Config, "FLY_MAX_VZ", 0.3),
            )
            thrust_kp = getattr(Config, "FLY_THRUST_KP", 0.35)
        elif phase == "hover":
            max_xy_speed = getattr(Config, "HOVER_MAX_XY_SPEED", 0.25)
            phase_kp_xy = Config.HOVER_POS_KP_XY
            brake_dist = getattr(Config, "HOVER_BRAKE_DIST_XY", 0.5)
            brake_decel = getattr(Config, "HOVER_BRAKE_DECEL", 1.2)
            brake_reverse_gain = getattr(Config, "HOVER_BRAKE_REVERSE_GAIN", 1.0)
            base_max_tilt = hover_max_a
            brake_max_tilt = max(
                base_max_tilt,
                getattr(Config, "HOVER_BRAKE_MAX_TILT_ANGLE", base_max_tilt),
            )
            desired_vz = torch.clamp(
                Config.HOVER_DESIRED_VZ_GAIN * tz,
                -Config.HOVER_MAX_VZ,
                Config.HOVER_MAX_VZ,
            )
            thrust_kp = Config.HOVER_THRUST_KP
        else:
            max_xy_speed = getattr(Config, "RISE_MAX_XY_SPEED", 0.0)
            phase_kp_xy = Config.RISE_POS_KP_XY
            brake_dist = 0.0
            brake_decel = 1.0
            brake_reverse_gain = 0.0
            base_max_tilt = rise_max_a
            brake_max_tilt = rise_max_a
            desired_vxy = self._clip_xy_vector(phase_kp_xy * target_rpos[:, :2], max_xy_speed)
            desired_vz = torch.clamp(
                Config.RISE_DESIRED_VZ_GAIN * tz,
                -Config.RISE_MAX_VZ,
                Config.RISE_MAX_VZ,
            )
            thrust_kp = Config.RISE_THRUST_KP

        if phase in ("fly", "hover"):
            (
                desired_vxy,
                brake_active,
                radial_speed_to_target,
                brake_speed_cap,
                stopping_dist_xy,
                desired_speed_xy,
            ) = self._compute_braked_desired_vxy(
                target_rpos[:, :2],
                v_world[:, :2],
                phase_kp_xy,
                max_xy_speed,
                brake_dist,
                brake_decel,
                brake_reverse_gain,
            )
            effective_tilt_limit = torch.where(
                brake_active,
                torch.full_like(dist_xy, brake_max_tilt),
                torch.full_like(dist_xy, base_max_tilt),
            )
        else:
            brake_active = torch.zeros_like(dist_xy, dtype=torch.bool)
            radial_speed_to_target = torch.zeros_like(dist_xy)
            brake_speed_cap = torch.zeros_like(dist_xy)
            stopping_dist_xy = torch.zeros_like(dist_xy)
            desired_speed_xy = torch.norm(desired_vxy, dim=-1)
            effective_tilt_limit = torch.full_like(dist_xy, rise_max_a)

        desired_velocity = torch.cat([desired_vxy, desired_vz.unsqueeze(-1)], dim=-1)
        vz_err = desired_vz - vz
        thrust = Config.HOVER_BASE_THRUST + thrust_kp * vz_err
        tilt_angle = torch.sqrt(current_roll**2 + current_pitch**2 + 1.0e-8)
        thrust = thrust / torch.clamp(torch.cos(tilt_angle), 0.7, 1.0)
        thrust = torch.clamp(thrust, -0.25, 0.25)

        target_body = torch.bmm(rotation.transpose(1, 2), target_rpos.unsqueeze(-1)).squeeze(-1)
        vel_err_world = desired_velocity - v_world
        vel_err_body = torch.bmm(rotation.transpose(1, 2), vel_err_world.unsqueeze(-1)).squeeze(-1)

        tilt_per_speed = torch.clamp(
            effective_tilt_limit / max(max_xy_speed, 1.0e-3),
            max=0.5,
        )

        d_roll_hold = torch.clamp(
            tilt_per_speed * vel_err_body[:, 1],
            -rise_max_a,
            rise_max_a,
        )
        d_pitch_hold = torch.clamp(
            -tilt_per_speed * vel_err_body[:, 0],
            -rise_max_a,
            rise_max_a,
        )

        d_roll_fly = torch.clamp(
            tilt_per_speed * vel_err_body[:, 1],
            -effective_tilt_limit,
            effective_tilt_limit,
        )
        d_pitch_fly = torch.clamp(
            -tilt_per_speed * vel_err_body[:, 0],
            -effective_tilt_limit,
            effective_tilt_limit,
        )

        hover_error_world = target_rpos
        hover_error_body = torch.bmm(rotation.transpose(1, 2), hover_error_world.unsqueeze(-1)).squeeze(-1)
        d_roll_hover = torch.clamp(
            tilt_per_speed * vel_err_body[:, 1],
            -effective_tilt_limit,
            effective_tilt_limit,
        )
        d_pitch_hover = torch.clamp(
            -tilt_per_speed * vel_err_body[:, 0],
            -effective_tilt_limit,
            effective_tilt_limit,
        )

        if phase == "fly":
            d_roll = d_roll_fly
            d_pitch = d_pitch_fly
        elif phase == "hover":
            d_roll = d_roll_hover
            d_pitch = d_pitch_hover
        else:
            d_roll = torch.zeros_like(current_roll)
            d_pitch = torch.zeros_like(current_pitch)

        k_rate = 2.0
        roll_rate = torch.clamp(k_rate * (d_roll - current_roll), -0.3, 0.3)
        pitch_rate = torch.clamp(k_rate * (d_pitch - current_pitch), -0.3, 0.3)

        action = torch.stack([roll_rate, pitch_rate, yaw_rate, thrust], dim=-1)
        debug = {
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "dist_xy": dist_xy,
            "yaw_error": yaw_error,
            "vx_b": vx_b,
            "vy_b": vy_b,
            "vz": vz,
            "hover_xy_speed": hover_xy_speed,
            "current_roll": current_roll,
            "current_pitch": current_pitch,
            "desired_vz": desired_vz,
            "vz_err": vz_err,
            "desired_vx_world": desired_velocity[:, 0],
            "desired_vy_world": desired_velocity[:, 1],
            "current_vx_world": v_world[:, 0],
            "current_vy_world": v_world[:, 1],
            "brake_active": brake_active,
            "radial_speed_to_target": radial_speed_to_target,
            "brake_speed_cap": brake_speed_cap,
            "stopping_dist_xy": stopping_dist_xy,
            "desired_speed_xy": desired_speed_xy,
            "vel_err_body_x": vel_err_body[:, 0],
            "vel_err_body_y": vel_err_body[:, 1],
            "hover_error_body_x": hover_error_body[:, 0],
            "hover_error_body_y": hover_error_body[:, 1],
            "d_roll_hover": d_roll_hover,
            "d_pitch_hover": d_pitch_hover,
            "roll_rate_cmd_sem": roll_rate,
            "pitch_rate_cmd_sem": pitch_rate,
            "target_body_x": target_body[:, 0],
            "target_body_y": target_body[:, 1],
        }
        return self._apply_action_axis_signs(action), debug

    # ------------------------------------------------------------------
    # Custom waypoint API
    # ------------------------------------------------------------------

    def set_custom_waypoints(self, offsets):
        """设置自定义途径点（相对于终点的偏移量）。

        设置后飞机将依次经过每个途径点，最后到达终点。
        每个途径点经历 上升→悬停→平飞 的完整流程。

        Args:
            offsets: list of [x, y, z]，例如 [[2, 0, -1], [1, 1, -2]]

        调用示例:
            controller.set_custom_waypoints([[2, 0, -1], [1, 1, -2]])
        """
        self._custom_wp_offsets = [list(o) for o in offsets]
        self.reset()

    def clear_custom_waypoints(self):
        """清除自定义途径点，恢复默认行为（避障规划或直飞终点）。"""
        self._custom_wp_offsets = None
        self.reset()

    # ------------------------------------------------------------------
    # Phase functions
    # ------------------------------------------------------------------

    def _phase_rise(self, parsed, target_rpos, mask):
        """阶段1：控制飞机升降+转向，升高到终点高度且机头朝向终点。

        完成后过渡到悬停阶段（或跳过悬停直接平飞）。
        Returns: (action, debug_dict)
        """
        action, debug = self._compute_tracking_action(parsed, target_rpos, phase="rise")

        rise_yaw_ok = debug["yaw_error"].abs() < 0.05
        h_ok = debug["tz"].abs() < Config.RISE_TO_HOVER_Z_THRESH
        overshot = debug["tz"] < -0.10
        rise_ready = mask & rise_yaw_ok & (h_ok | overshot)
        bypass_hover = rise_ready & (
            debug["dist_xy"] > getattr(Config, "HOVER_BYPASS_DIST_XY", float("inf"))
        )
        self._wp_phase[bypass_hover] = 2
        self._hover_timer[bypass_hover] = 0
        self._hover_ref_target_rpos[bypass_hover].zero_()

        to_hover = rise_ready & (~bypass_hover)
        self._wp_phase[to_hover] = 1
        self._hover_timer[to_hover] = getattr(Config, "HOVER_ENTRY_HOLD_FRAMES", 125)
        self._hover_ref_target_rpos[to_hover] = target_rpos[to_hover]

        return action, debug

    def _phase_hover(self, parsed, target_rpos, mask):
        """阶段2：悬停0.9s，稳定姿态。

        完成后过渡到平飞阶段。
        Returns: (action, debug_dict)
        """
        action, debug = self._compute_tracking_action(
            parsed,
            target_rpos,
            phase="hover",
            hover_ref_target_rpos=self._hover_ref_target_rpos,
        )

        self._hover_timer[mask] -= 1
        hover_yaw_ok = debug["yaw_error"].abs() < 0.05
        hover_xy_ok = debug["hover_xy_speed"] < getattr(
            Config, "HOVER_STABLE_XY_SPEED", float("inf")
        )
        stable_hover = (
            mask
            & hover_yaw_ok
            & (debug["tz"].abs() < Config.HOVER_STABLE_Z_THRESH)
            & hover_xy_ok
        )
        unstable_hover = mask & (~stable_hover) & (~self._wp_from_fly)
        self._hover_timer[unstable_hover] = torch.clamp(
            self._hover_timer[unstable_hover],
            min=getattr(Config, "HOVER_UNSTABLE_MIN_FRAMES", 60),
        )
        brake_done = self._wp_from_fly & (self._hover_timer <= 0)
        self._wp_phase[brake_done] = 0
        self._wp_from_fly[brake_done] = False

        to_fly = stable_hover & (self._hover_timer <= 0) & (~self._wp_from_fly)
        self._wp_phase[to_fly] = 2

        debug["hover_xy_stable"] = hover_xy_ok
        return action, debug

    def _phase_fly(self, parsed, target_rpos, mask):
        """阶段3：控制飞机平飞到终点。

        到达终点后过渡到最终悬停；到达中间航点则前进到下一航点。
        Returns: (action, debug_dict)
        """
        action, debug = self._compute_tracking_action(parsed, target_rpos, phase="fly")

        batch_size = parsed.batch_size
        device = parsed.device

        is_goal_target = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for b in range(batch_size):
            offsets = self._wp_offsets[b]
            idx = int(self._wp_index[b].item())
            is_goal_target[b] = offsets is None or idx >= len(offsets)

        wp_xy_ok = debug["dist_xy"] < getattr(Config, "ROUTE_WP_REACHED_DIST_XY", 0.3)
        wp_z_ok = debug["tz"].abs() < getattr(Config, "ROUTE_WP_REACHED_DIST_Z", 0.3)
        wp_speed_ok = debug["hover_xy_speed"] < getattr(Config, "ROUTE_WP_REACHED_SPEED_XY", 0.2)
        goal_xy_ok = debug["dist_xy"] < getattr(Config, "GOAL_HOVER_DIST_XY", 0.2)
        goal_z_ok = debug["tz"].abs() < getattr(Config, "GOAL_HOVER_DIST_Z", 0.2)
        goal_speed_ok = debug["hover_xy_speed"] < getattr(Config, "GOAL_HOVER_SPEED_XY", 0.15)

        to_final_hover = mask & is_goal_target & goal_xy_ok & goal_z_ok & goal_speed_ok
        self._wp_phase[to_final_hover] = 3
        self._hover_timer[to_final_hover] = 0
        self._hover_ref_target_rpos[to_final_hover] = target_rpos[to_final_hover]

        to_next = mask & (~is_goal_target) & wp_xy_ok & wp_z_ok & wp_speed_ok
        if to_next.any():
            for b in range(batch_size):
                if not to_next[b]:
                    continue
                offsets = self._wp_offsets[b]
                idx = int(self._wp_index[b].item())
                if idx < len(offsets):
                    self._wp_index[b] = idx + 1
                    self._wp_phase[b] = 1
                    self._hover_timer[b] = 50
                    self._wp_from_fly[b] = True
                    self._hover_ref_target_rpos[b] = target_rpos[b].clone()

        debug["is_goal_target"] = is_goal_target
        return action, debug

    def _phase_final(self, parsed, target_rpos, mask):
        """阶段4：到终点后悬停3s。无退出过渡，复用悬停控制律。

        Returns: (None, debug_dict)
        """
        return None, {"is_final": mask}

    def _compute_action_full(self, obs) -> torch.Tensor:
        obs, original_shape = self._prepare_obs(obs)
        parsed = self.parser.parse(obs)
        batch_size = parsed.batch_size
        device = parsed.device

        if self._wp_offsets is None or len(self._wp_offsets) != batch_size:
            self._init_state(batch_size, device)

        goal_rpos = parsed.goal_rpos
        for b in range(batch_size):
            if self._wp_offsets[b] is not None:
                continue

            if self._custom_wp_offsets is not None:
                self._wp_offsets[b] = list(self._custom_wp_offsets)
            else:
                waypoints = []
                g = parsed.goal_rpos[b]
                for i in range(8):
                    if parsed.waypoint_active[b, i] and not parsed.waypoint_visited[b, i]:
                        waypoints.append(parsed.waypoint_rpos[b, i])

                waypoints.sort(key=lambda w: (w**2).sum().item())

                waypoint_offsets = []
                for wp in waypoints:
                    waypoint_offsets.append(
                        [(wp[0] - g[0]).item(), (wp[1] - g[1]).item(), (wp[2] - g[2]).item()]
                    )
                self._wp_offsets[b] = waypoint_offsets

            self._wp_index[b] = 0
            self._wp_phase[b] = 0
            self._hover_timer[b] = 0
            self._hover_ref_target_rpos[b].zero_()

            if b == 0:
                n = len(self._wp_offsets[b])
                if self._custom_wp_offsets is not None:
                    msg = (
                        f"[CUSTOM-WP] {n} custom waypoints set, "
                        f"will fly through them before reaching goal."
                    )
                elif n > 0:
                    msg = f"[ENV-WP] found {n} active waypoints from observation"
                elif self._direct_goal_mode:
                    msg = (
                        "[DIRECT-GOAL] single-target mode active: "
                        "ignore waypoints, fly goal directly."
                    )
                else:
                    msg = "[ENV-WP] no active waypoints, using direct path"
                if self.logger:
                    self.logger.info(msg)
                print(msg, flush=True)

        target_rpos = torch.zeros(batch_size, 3, device=device)
        for b in range(batch_size):
            offsets = self._wp_offsets[b]
            idx = int(self._wp_index[b].item())
            if offsets is None or idx > len(offsets):
                target_rpos[b] = goal_rpos[b]
            elif idx < len(offsets):
                offset = torch.tensor(offsets[idx], dtype=torch.float32, device=device)
                target_rpos[b] = goal_rpos[b] + offset
            else:
                target_rpos[b] = goal_rpos[b]

        # 制动悬停：目标设为零向量，原地刹车
        target_rpos[self._wp_from_fly] = 0.0

        mask_rise = self._wp_phase == 0
        mask_hover = self._wp_phase == 1
        mask_fly = self._wp_phase == 2
        mask_final = self._wp_phase == 3

        action_rise, rise_debug = self._phase_rise(parsed, target_rpos, mask_rise)
        action_hover, hover_debug = self._phase_hover(parsed, target_rpos, mask_hover)
        action_fly, fly_debug = self._phase_fly(parsed, target_rpos, mask_fly)
        _, final_debug = self._phase_final(parsed, target_rpos, mask_final)

        action = torch.where(
            mask_fly.unsqueeze(-1),
            action_fly,
            torch.where((mask_hover | mask_final).unsqueeze(-1), action_hover, action_rise),
        )

        if not hasattr(self, "_full_fc"):
            self._full_fc = 0
        self._full_fc += 1
        if self._full_fc <= 15 or self._full_fc % 30 == 0:
            phase_names = {0: "RISE", 1: "HOVER", 2: "FLY", 3: "FINAL"}
            b0 = 0
            wp_idx = int(self._wp_index[b0].item())
            n_wp = len(self._wp_offsets[b0]) if self._wp_offsets[b0] else 0
            p = int(self._wp_phase[b0].item())
            phase_debug = {0: rise_debug, 1: hover_debug, 2: fly_debug, 3: hover_debug}.get(p, fly_debug)
            msg = (
                f">>> 航路飞行 f{self._full_fc} [{phase_names.get(p, '?')}] "
                f"wp={wp_idx}/{n_wp} "
                f"t=({phase_debug['tx'][b0]:.2f},{phase_debug['ty'][b0]:.2f},{phase_debug['tz'][b0]:.2f}) "
                f"d_xy={phase_debug['dist_xy'][b0]:.2f} yaw_e={phase_debug['yaw_error'][b0]:.2f} "
                f"h_timer={self._hover_timer[b0].item()} "
                f"act=({action[b0,0]:.2f},{action[b0,1]:.2f},{action[b0,2]:.2f},{action[b0,3]:.3f})"
            )
            if p in (1, 3):
                msg += (
                    f" hover_err_b=({hover_debug['hover_error_body_x'][b0]:+.2f},"
                    f"{hover_debug['hover_error_body_y'][b0]:+.2f}) "
                    f"v_b=({hover_debug['vx_b'][b0]:+.2f},{hover_debug['vy_b'][b0]:+.2f},"
                    f"{hover_debug['vz'][b0]:+.2f}) "
                    f"xy_v={hover_debug['hover_xy_speed'][b0]:.2f} "
                    f"stable_xy={int(hover_debug['hover_xy_stable'][b0].item())} "
                    f"att=({hover_debug['current_roll'][b0]:+.2f},"
                    f"{hover_debug['current_pitch'][b0]:+.2f}) "
                    f"rr_sem={hover_debug['roll_rate_cmd_sem'][b0]:+.2f} "
                    f"pr_sem={hover_debug['pitch_rate_cmd_sem'][b0]:+.2f} "
                    f"d_hover=({hover_debug['d_roll_hover'][b0]:+.2f},"
                    f"{hover_debug['d_pitch_hover'][b0]:+.2f})"
                )
            elif p == 2:
                msg += (
                    f" v_b=({fly_debug['vx_b'][b0]:+.2f},{fly_debug['vy_b'][b0]:+.2f},"
                    f"{fly_debug['vz'][b0]:+.2f}) "
                    f" v_w=({fly_debug['current_vx_world'][b0]:+.2f},{fly_debug['current_vy_world'][b0]:+.2f}) "
                    f" v_des=({fly_debug['desired_vx_world'][b0]:+.2f},{fly_debug['desired_vy_world'][b0]:+.2f}) "
                    f" brake={int(fly_debug['brake_active'][b0].item())} "
                    f"vr={fly_debug['radial_speed_to_target'][b0]:+.2f} "
                    f"v_cap={fly_debug['brake_speed_cap'][b0]:+.2f} "
                    f"stop_d={fly_debug['stopping_dist_xy'][b0]:+.2f} "
                    f"att=({fly_debug['current_roll'][b0]:+.2f},"
                    f"{fly_debug['current_pitch'][b0]:+.2f}) "
                    f"rr_sem={fly_debug['roll_rate_cmd_sem'][b0]:+.2f} "
                    f"pr_sem={fly_debug['pitch_rate_cmd_sem'][b0]:+.2f} "
                    f"vz_cmd={fly_debug['desired_vz'][b0]:+.2f} "
                    f"vz_err={fly_debug['vz_err'][b0]:+.2f}"
                )
            if self.logger:
                self.logger.info(msg)
            print(msg, flush=True)

        return self._restore_shape(action, original_shape)

    def _log_plan(self, b, parsed, offsets, total):
        goal = parsed.goal_rpos[b]
        gx, gy, gz = goal[0].item(), goal[1].item(), goal[2].item()

        lines = []
        lines.append("=" * 60)
        lines.append(f"[WP-PLAN b{b}]  full plan report")

        active_mask = parsed.obstacle_active[b]
        n_obs = active_mask.sum().item()
        lines.append(f"  active obstacles: {n_obs}")
        obs_list = []
        for i in range(len(active_mask)):
            if active_mask[i]:
                ox = parsed.obstacle_rpos[b, i, 0].item()
                oy = parsed.obstacle_rpos[b, i, 1].item()
                oz = parsed.obstacle_rpos[b, i, 2].item()
                r = parsed.obstacle_radius[b, i].item()
                lines.append(f"    obs[{i}]: pos=({ox:+.2f},{oy:+.2f},{oz:+.2f}) r={r:.2f}")
                obs_list.append({"pos": (ox, oy, oz), "radius": r})

        lines.append(f"  total targets (incl. goal): {total}")
        lines.append(f"  goal rpos: ({gx:+.2f},{gy:+.2f},{gz:+.2f})")

        if not offsets:
            lines.append("  direct path - no waypoints")
            min_d = self._min_clearance_to_path([0.0, 0.0, 0.0], [gx, gy, gz], obs_list)
            lines.append(
                f"  direct clearance: {min_d:.3f}m "
                f"(need >{self.wp_planner._safety_radius:.2f}m)"
            )
        else:
            wp_now = self.wp_planner.waypoint_rpos_now(goal, offsets)
            lines.append(f"  waypoints ({len(offsets)}):")
            for i, (wp, off) in enumerate(zip(wp_now, offsets)):
                lines.append(
                    f"    WP{i}: rpos=({wp[0]:+.2f},{wp[1]:+.2f},{wp[2]:+.2f})  "
                    f"offset=({off[0]:+.2f},{off[1]:+.2f},{off[2]:+.2f})"
                )

            start = [0.0, 0.0, 0.0]
            all_nodes = [start] + [[w[0], w[1], w[2]] for w in wp_now] + [[gx, gy, gz]]
            lines.append(f"  segments ({len(all_nodes)-1}):")
            total_len = 0.0
            for i in range(len(all_nodes) - 1):
                a = all_nodes[i]
                bnode = all_nodes[i + 1]
                seg_len = math.sqrt(
                    (bnode[0] - a[0]) ** 2 + (bnode[1] - a[1]) ** 2 + (bnode[2] - a[2]) ** 2
                )
                total_len += seg_len
                min_d = self._min_clearance_to_path(a, bnode, obs_list)
                label = {0: "start"}.get(i, f"WP{i-1}")
                label_next = f"WP{i}" if i < len(all_nodes) - 2 else "goal"
                lines.append(
                    f"    {label}->{label_next}: "
                    f"({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f})->"
                    f"({bnode[0]:+.2f},{bnode[1]:+.2f},{bnode[2]:+.2f}) "
                    f"len={seg_len:.2f}m clr={min_d:.3f}m"
                )
            lines.append(f"  total path length: {total_len:.2f}m")
            lines.append(f"  direct distance: {math.sqrt(gx*gx + gy*gy + gz*gz):.2f}m")

        lines.append("=" * 60)
        msg = "\n".join(lines)
        if self.logger:
            self.logger.info(msg)
        print(msg, flush=True)

    @staticmethod
    def _min_clearance_to_path(a, b, obstacles):
        if not obstacles:
            return float("inf")
        min_d = float("inf")
        flight_z = b[2]
        for obs in obstacles:
            oz = obs["pos"][2]
            r = obs["radius"]
            if abs(flight_z - oz) >= r + 0.05:
                continue
            abx = b[0] - a[0]
            aby = b[1] - a[1]
            if abs(abx) < 1.0e-9 and abs(aby) < 1.0e-9:
                d = math.hypot(obs["pos"][0] - a[0], obs["pos"][1] - a[1])
            else:
                t = ((obs["pos"][0] - a[0]) * abx + (obs["pos"][1] - a[1]) * aby) / (abx * abx + aby * aby)
                t = max(0.0, min(1.0, t))
                px = a[0] + t * abx
                py = a[1] + t * aby
                d = math.hypot(obs["pos"][0] - px, obs["pos"][1] - py)
            if d < min_d:
                min_d = d
        return min_d

    def _init_state(self, batch_size: int, device: torch.device):
        self._target_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._needs_replan = torch.ones(batch_size, dtype=torch.bool, device=device)
        self._wp_offsets = [None] * batch_size
        self._wp_index = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._wp_phase = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._hover_timer = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._hover_ref_target_rpos = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
        self._wp_from_fly = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def reset(self, batch_size: int = None, device: torch.device = None):
        if batch_size is not None and device is not None:
            self._init_state(batch_size, device)
        else:
            self._target_idx = None
            self._needs_replan = None
            self._wp_offsets = None
            self._wp_index = None
            self._wp_phase = None
            self._hover_timer = None
            self._hover_ref_target_rpos = None
            self._wp_from_fly = None


__all__ = ["HierarchicalController"]
