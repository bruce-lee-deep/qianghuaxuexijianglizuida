#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Parse the 95D ObstacleHover observation tensor into structured components.
All positions are relative (rpos = reference_point - drone_position).
无人机位置不在观测中，所有位置均为相对位置。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np

from agent_ppo.conf.conf import Config


@dataclass
class ParsedObservation:
    """Structured observation extracted from the 95D flat tensor.

    All positions are RELATIVE to the drone (rpos = target - drone).
    The drone itself is at (0, 0, 0) in this relative frame.
    """

    batch_size: int
    device: torch.device

    # Target / Goal (both are the same: goal_position - drone_position)
    target_rpos: torch.Tensor     # [batch, 3]
    goal_rpos: torch.Tensor       # [batch, 3]  (= target_rpos)

    # Start (start_position - drone_position)
    start_rpos: torch.Tensor      # [batch, 3]

    # Obstacles (relative positions + radii)
    obstacle_rpos: torch.Tensor   # [batch, max_obs, 3]
    obstacle_radius: torch.Tensor # [batch, max_obs]
    obstacle_active: torch.Tensor # [batch, max_obs] bool

    # Waypoints (relative positions + visited flags)
    waypoint_rpos: torch.Tensor      # [batch, max_wp, 3]
    waypoint_visited: torch.Tensor   # [batch, max_wp] bool
    waypoint_active: torch.Tensor    # [batch, max_wp] bool

    # Drone velocity (body frame)
    linear_velocity: torch.Tensor    # [batch, 3]  body-frame m/s
    angular_velocity: torch.Tensor   # [batch, 3]  body-frame rad/s

    # Rotation matrix (body → world)
    rotation_matrix: torch.Tensor    # [batch, 3, 3]

    # Phase
    phase_nav: torch.Tensor          # [batch] 1.0=nav
    phase_hover: torch.Tensor        # [batch] 1.0=hover

    # Time encoding
    time_encoding: torch.Tensor      # [batch, 4]

    # Arena bounds
    arena_bounds: dict


class ObservationParser:
    """Parse a flat 95D observation into a ParsedObservation.

    Layout per official documentation:
        0-2:   target_rpos (goal - drone)
        3-26:  obstacle_rpos (8 × 3)
        27-29: linear_velocity (body frame)
        30-32: angular_velocity (body frame)
        33-41: rotation_matrix (3×3 flattened)
        42:    hover_timer
        43-50: obstacle_radii (8 × 1)
        51-53: start_rpos
        54-56: goal_rpos (= target_rpos)
        57-58: phase_onehot
        59-82: waypoint_rpos (8 × 3)
        83-90: waypoint_visited (8 × 1)
        91-94: time_encoding
    """

    def __init__(self):
        self._cfg = Config

    def parse(self, obs) -> ParsedObservation:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        if obs.dim() == 3:
            obs = obs.view(obs.shape[0] * obs.shape[1], -1)
        if obs.dim() != 2 or obs.shape[-1] != Config.OBS_DIM:
            raise ValueError(
                f"Expected obs shape [*, {Config.OBS_DIM}], got {tuple(obs.shape)}"
            )

        batch_size = obs.shape[0]
        device = obs.device
        cfg = Config

        # --- Relative positions ---
        target_rpos = obs[:, cfg.OBS_IDX_TARGET_RPOS]
        goal_rpos = obs[:, cfg.OBS_IDX_GOAL_RPOS]
        start_rpos = obs[:, cfg.OBS_IDX_START_RPOS]

        # --- Velocity (body frame) ---
        linear_velocity = obs[:, cfg.OBS_IDX_LINEAR_VELOCITY]
        angular_velocity = obs[:, cfg.OBS_IDX_ANGULAR_VELOCITY]

        # --- Rotation matrix ---
        r_start = cfg.OBS_IDX_ROTATION_MATRIX_START
        rot_flat = obs[:, r_start:r_start + 9]
        rotation_matrix = rot_flat.view(batch_size, 3, 3)

        # --- Phase ---
        phase_onehot = obs[:, cfg.OBS_IDX_PHASE_ONEHOT]
        phase_nav = phase_onehot[:, 0]    # > 0.5 = navigation
        phase_hover = phase_onehot[:, 1]  # > 0.5 = hover

        # --- Obstacles (8 slots, each 3D rpos + 1D radius) ---
        obs_max = Config.OBSTACLE_MAX_COUNT
        obs_rpos_start = cfg.OBS_IDX_OBSTACLE_RPOS_START
        obs_rad_start = cfg.OBS_IDX_OBSTACLE_RADII_START

        obstacle_rpos = torch.zeros(batch_size, obs_max, 3, device=device)
        obstacle_radius = torch.zeros(batch_size, obs_max, device=device)
        obstacle_active = torch.zeros(batch_size, obs_max, dtype=torch.bool, device=device)

        for i in range(obs_max):
            rp = obs_rpos_start + i * 3
            obstacle_rpos[:, i, 0] = obs[:, rp]
            obstacle_rpos[:, i, 1] = obs[:, rp + 1]
            obstacle_rpos[:, i, 2] = obs[:, rp + 2]
            obstacle_radius[:, i] = obs[:, obs_rad_start + i]
            # Active if radius > 0 (per docs: 未激活为 0)
            obstacle_active[:, i] = obstacle_radius[:, i] > 1e-6

        # --- Waypoints (8 slots, each 3D rpos + 1D visited) ---
        wp_max = Config.MAX_WAYPOINTS
        wp_rpos_start = cfg.OBS_IDX_WAYPOINT_RPOS_START
        wp_vis_start = cfg.OBS_IDX_WAYPOINT_VISITED_START

        waypoint_rpos = torch.zeros(batch_size, wp_max, 3, device=device)
        waypoint_visited = torch.zeros(batch_size, wp_max, dtype=torch.bool, device=device)
        waypoint_active = torch.zeros(batch_size, wp_max, dtype=torch.bool, device=device)

        for i in range(wp_max):
            rp = wp_rpos_start + i * 3
            waypoint_rpos[:, i, 0] = obs[:, rp]
            waypoint_rpos[:, i, 1] = obs[:, rp + 1]
            waypoint_rpos[:, i, 2] = obs[:, rp + 2]
            vis_val = obs[:, wp_vis_start + i]
            waypoint_visited[:, i] = vis_val > 0.5
            # Active if rpos norm > 0 (unused slots are zeroed)
            waypoint_active[:, i] = waypoint_rpos[:, i].norm(dim=-1) > 1e-6

        # --- Time encoding ---
        time_encoding = obs[:, cfg.OBS_IDX_TIME_ENCODING]

        # --- Arena bounds ---
        arena_bounds = {
            "x_min": Config.ARENA_X_MIN, "x_max": Config.ARENA_X_MAX,
            "y_min": Config.ARENA_Y_MIN, "y_max": Config.ARENA_Y_MAX,
            "z_min": Config.ARENA_Z_MIN, "z_max": Config.ARENA_Z_MAX,
        }

        return ParsedObservation(
            batch_size=batch_size, device=device,
            target_rpos=target_rpos, goal_rpos=goal_rpos, start_rpos=start_rpos,
            obstacle_rpos=obstacle_rpos, obstacle_radius=obstacle_radius,
            obstacle_active=obstacle_active,
            waypoint_rpos=waypoint_rpos, waypoint_visited=waypoint_visited,
            waypoint_active=waypoint_active,
            linear_velocity=linear_velocity, angular_velocity=angular_velocity,
            rotation_matrix=rotation_matrix,
            phase_nav=phase_nav, phase_hover=phase_hover,
            time_encoding=time_encoding, arena_bounds=arena_bounds,
        )


__all__ = ["ObservationParser", "ParsedObservation"]
