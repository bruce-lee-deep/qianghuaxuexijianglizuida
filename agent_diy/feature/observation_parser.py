#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agent_diy.conf.conf import Config


@dataclass
class ParsedObservation:
    batch_size: int
    device: torch.device
    target_rpos: torch.Tensor
    goal_rpos: torch.Tensor
    start_rpos: torch.Tensor
    waypoint_rpos: torch.Tensor
    waypoint_visited: torch.Tensor
    waypoint_active: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    rotation_matrix: torch.Tensor
    current_rpy: torch.Tensor
    phase_onehot: torch.Tensor
    time_encoding: torch.Tensor


class ObservationParser:
    def parse(self, obs) -> ParsedObservation:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        if obs.dim() == 3:
            obs = obs.view(obs.shape[0] * obs.shape[1], -1)
        if obs.dim() != 2 or obs.shape[-1] != Config.OBS_DIM:
            raise ValueError(f"Expected obs shape [*, {Config.OBS_DIM}], got {tuple(obs.shape)}")

        batch_size = obs.shape[0]
        device = obs.device

        target_rpos = obs[:, Config.OBS_IDX_TARGET_RPOS]
        goal_rpos = obs[:, Config.OBS_IDX_GOAL_RPOS]
        start_rpos = obs[:, Config.OBS_IDX_START_RPOS]
        linear_velocity = obs[:, Config.OBS_IDX_LINEAR_VELOCITY]
        angular_velocity = obs[:, Config.OBS_IDX_ANGULAR_VELOCITY]

        r_start = Config.OBS_IDX_ROTATION_MATRIX_START
        rotation_matrix = obs[:, r_start : r_start + 9].view(batch_size, 3, 3)

        r20 = torch.clamp(rotation_matrix[:, 2, 0], -0.999, 0.999)
        pitch = torch.asin(-r20)
        roll = torch.atan2(rotation_matrix[:, 2, 1], rotation_matrix[:, 2, 2])
        yaw = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
        current_rpy = torch.stack([roll, pitch, yaw], dim=-1)

        phase_onehot = obs[:, Config.OBS_IDX_PHASE_ONEHOT]
        time_encoding = obs[:, Config.OBS_IDX_TIME_ENCODING]

        wp_max = Config.MAX_WAYPOINTS
        wp_rpos = torch.zeros(batch_size, wp_max, 3, device=device)
        wp_visited = torch.zeros(batch_size, wp_max, dtype=torch.bool, device=device)
        wp_active = torch.zeros(batch_size, wp_max, dtype=torch.bool, device=device)

        for i in range(wp_max):
            rp = Config.OBS_IDX_WAYPOINT_RPOS_START + i * 3
            wp_rpos[:, i, 0] = obs[:, rp]
            wp_rpos[:, i, 1] = obs[:, rp + 1]
            wp_rpos[:, i, 2] = obs[:, rp + 2]
            wp_visited[:, i] = obs[:, Config.OBS_IDX_WAYPOINT_VISITED_START + i] > 0.5
            wp_active[:, i] = torch.norm(wp_rpos[:, i], dim=-1) > 1.0e-6

        return ParsedObservation(
            batch_size=batch_size,
            device=device,
            target_rpos=target_rpos,
            goal_rpos=goal_rpos,
            start_rpos=start_rpos,
            waypoint_rpos=wp_rpos,
            waypoint_visited=wp_visited,
            waypoint_active=wp_active,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            rotation_matrix=rotation_matrix,
            current_rpy=current_rpy,
            phase_onehot=phase_onehot,
            time_encoding=time_encoding,
        )


__all__ = ["ObservationParser", "ParsedObservation"]
