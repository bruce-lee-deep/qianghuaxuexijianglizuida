#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Potential-field-based path planner for obstacle avoidance.
基于人工势场法的障碍物避碰路径规划器。
"""

from __future__ import annotations

import torch

from agent_ppo.conf.conf import Config


class PotentialFieldPlanner:
    """Compute a desired velocity that avoids obstacles and moves toward a target.

    The total virtual force on the drone is:
        F_total = F_att(target) + F_rep(obstacles) + F_rep(boundaries) - k_damp * v

    The desired velocity is proportional to F_total, clipped to max_velocity.

    Usage:
        planner = PotentialFieldPlanner()
        v_des = planner.compute_desired_velocity(drone_pos, drone_vel, target_pos,
                                                  obstacle_pos, obstacle_radius,
                                                  obstacle_active, arena_bounds)
    """

    def __init__(self):
        cfg = Config
        self.k_att = cfg.PF_ATTRACTIVE_GAIN
        self.k_rep = cfg.PF_REPULSIVE_GAIN
        self.d0 = cfg.PF_REPULSIVE_RANGE
        self.k_damp = cfg.PF_DAMPING_GAIN
        self.max_vel = cfg.PF_MAX_VELOCITY
        self.k_bound = cfg.PF_BOUNDARY_REPULSIVE_GAIN
        self.boundary_margin = cfg.PF_BOUNDARY_MARGIN
        self.safety_radius = cfg.SAFETY_RADIUS

    def compute_desired_velocity(
        self,
        drone_pos: torch.Tensor,          # [batch, 3]
        drone_vel: torch.Tensor,          # [batch, 3]
        target_pos: torch.Tensor,         # [batch, 3]
        obstacle_pos: torch.Tensor,       # [batch, max_obs, 2]  xy only
        obstacle_radius: torch.Tensor,    # [batch, max_obs]
        obstacle_active: torch.Tensor,    # [batch, max_obs] bool
        arena_bounds: dict,
    ) -> torch.Tensor:
        """Return desired velocity [batch, 3] in world frame."""
        batch_size = drone_pos.shape[0]
        device = drone_pos.device

        # --- Attractive force toward target ---
        to_target = target_pos - drone_pos  # [batch, 3]
        dist_to_target = torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-3)
        f_att = self.k_att * to_target / dist_to_target  # Constant magnitude away from origin
        # Scale down when very close to avoid overshoot.
        close_mask = (dist_to_target < 1.0).float()
        f_att = f_att * (close_mask * dist_to_target + (1 - close_mask))

        # --- Repulsive force from obstacles ---
        f_rep = torch.zeros_like(drone_pos)

        # We only have obstacle xy, so compute repulsion in xy plane.
        drone_xy = drone_pos[:, :2]  # [batch, 2]
        max_obs = obstacle_pos.shape[1]

        for i in range(max_obs):
            obs_xy = obstacle_pos[:, i, :]  # [batch, 2]
            obs_r = obstacle_radius[:, i]    # [batch]
            active = obstacle_active[:, i]   # [batch] bool

            to_obs = drone_xy - obs_xy       # [batch, 2]
            dist_xy = torch.norm(to_obs, dim=-1).clamp(min=1e-4)  # [batch]

            # Effective distance = xy distance - obstacle radius - drone radius.
            drone_r = 0.05
            effective_dist = dist_xy - obs_r - drone_r

            # Repulsive force only within range d0 and for active obstacles.
            in_range = (effective_dist < self.d0) & active  # [batch] bool

            if in_range.any():
                # F_rep = k_rep * (1/d_eff - 1/d0) / d_eff^2 * direction
                d_eff_clamped = effective_dist.clamp(min=1e-3)
                magnitude = (
                    self.k_rep
                    * (1.0 / d_eff_clamped - 1.0 / self.d0)
                    / (d_eff_clamped * d_eff_clamped)
                )
                direction = to_obs / dist_xy.unsqueeze(-1).clamp(min=1e-4)
                force_xy = magnitude.unsqueeze(-1) * direction * in_range.float().unsqueeze(-1)
                f_rep[:, :2] = f_rep[:, :2] + force_xy

        # --- Repulsive force from arena boundaries ---
        f_bound = torch.zeros_like(drone_pos)
        margin = self.boundary_margin

        # X boundaries
        x_min = arena_bounds["x_min"]
        x_max = arena_bounds["x_max"]
        dist_x_min = drone_pos[:, 0] - x_min
        dist_x_max = x_max - drone_pos[:, 0]

        push_x_min = (dist_x_min < margin).float() * self.k_bound * (margin - dist_x_min).clamp(min=0)
        push_x_max = (dist_x_max < margin).float() * self.k_bound * (margin - dist_x_max).clamp(min=0)
        f_bound[:, 0] = push_x_min - push_x_max

        # Y boundaries
        y_min = arena_bounds["y_min"]
        y_max = arena_bounds["y_max"]
        dist_y_min = drone_pos[:, 1] - y_min
        dist_y_max = y_max - drone_pos[:, 1]

        push_y_min = (dist_y_min < margin).float() * self.k_bound * (margin - dist_y_min).clamp(min=0)
        push_y_max = (dist_y_max < margin).float() * self.k_bound * (margin - dist_y_max).clamp(min=0)
        f_bound[:, 1] = push_y_min - push_y_max

        # Z boundaries
        z_min = arena_bounds["z_min"]
        z_max = arena_bounds["z_max"]
        dist_z_min = drone_pos[:, 2] - z_min
        dist_z_max = z_max - drone_pos[:, 2]

        push_z_min = (dist_z_min < margin).float() * self.k_bound * (margin - dist_z_min).clamp(min=0)
        push_z_max = (dist_z_max < margin).float() * self.k_bound * (margin - dist_z_max).clamp(min=0)
        f_bound[:, 2] = push_z_min - push_z_max

        f_rep = f_rep + f_bound

        # --- Net force → desired velocity ---
        f_total = f_att + f_rep - self.k_damp * drone_vel

        # Convert force to velocity (simplified: a = F/m, m=1, v_des = v + a*dt)
        # For direct velocity command: v_des proportional to net force
        desired_vel = f_total

        # Clamp velocity.
        vel_norm = torch.norm(desired_vel, dim=-1, keepdim=True)
        scale = torch.where(
            vel_norm > self.max_vel,
            self.max_vel / vel_norm.clamp(min=1e-4),
            torch.ones_like(vel_norm),
        )
        desired_vel = desired_vel * scale

        return desired_vel

    def check_emergency(
        self,
        drone_pos: torch.Tensor,
        obstacle_pos: torch.Tensor,
        obstacle_radius: torch.Tensor,
        obstacle_active: torch.Tensor,
    ) -> torch.Tensor:
        """Check if any env is in emergency (too close to an obstacle).

        Returns [batch] bool tensor.
        """
        drone_xy = drone_pos[:, :2]
        max_obs = obstacle_pos.shape[1]
        min_dist = torch.full((drone_pos.shape[0],), float("inf"), device=drone_pos.device)

        for i in range(max_obs):
            obs_xy = obstacle_pos[:, i, :]
            obs_r = obstacle_radius[:, i]
            active = obstacle_active[:, i]
            dist_xy = torch.norm(drone_xy - obs_xy, dim=-1)
            effective = dist_xy - obs_r - 0.05  # 0.05 = drone_radius
            min_dist = torch.min(min_dist, effective)

        return min_dist < self.safety_radius

    def emergency_action(
        self,
        drone_pos: torch.Tensor,
        drone_vel: torch.Tensor,
        obstacle_pos: torch.Tensor,
        obstacle_radius: torch.Tensor,
        obstacle_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate emergency evasion velocity and yaw_rate.

        Returns:
            emergency_vel: [batch, 3] — velocity away from nearest obstacle.
            emergency_yaw_rate: [batch] — zero (maintain heading during evasion).
        """
        batch_size = drone_pos.shape[0]
        device = drone_pos.device
        drone_xy = drone_pos[:, :2]

        # Find nearest obstacle.
        nearest_dir = torch.zeros(batch_size, 2, device=device)
        nearest_dir[:, 0] = 1.0  # Default: forward

        for b in range(batch_size):
            best_dist = float("inf")
            for i in range(obstacle_pos.shape[1]):
                if not obstacle_active[b, i]:
                    continue
                obs_xy = obstacle_pos[b, i, :]
                obs_r = obstacle_radius[b, i]
                to_obs = drone_xy[b] - obs_xy
                dist = torch.norm(to_obs) - obs_r - 0.05
                if dist < best_dist:
                    best_dist = dist
                    if torch.norm(to_obs) > 1e-6:
                        nearest_dir[b] = to_obs / torch.norm(to_obs)

        # Evasion: backward + up (combine for 3D avoidance).
        evasion_xy = nearest_dir * Config.EMERGENCY_BACKWARD_SPEED  # away from obs
        evasion_z = torch.ones(batch_size, device=device) * Config.EMERGENCY_CLIMB_SPEED

        emergency_vel = torch.cat([evasion_xy, evasion_z.unsqueeze(-1)], dim=-1)
        emergency_yaw_rate = torch.zeros(batch_size, device=device)

        return emergency_vel, emergency_yaw_rate


__all__ = ["PotentialFieldPlanner"]
