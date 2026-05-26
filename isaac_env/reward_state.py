#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Read-only reward state snapshot for ObstacleHover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _clone_optional_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return _clone_tensor(value)


@dataclass(frozen=True)
class RewardStateSnapshot:
    """Immutable snapshot consumed by RewardProcess.

    The snapshot is rebuilt from the live env on each step. All tensor fields are
    cloned so reward-side mutations cannot flow back into the simulator state.
    """

    device: torch.device
    num_envs: int
    nav_phase: int
    hover_phase: int
    arrival_radius: float
    stall_progress_threshold: float
    nav_max_steps: int
    hover_max_steps: int
    drone_radius: float
    obstacle_safe_distance: float
    waypoint_collect_radius: float
    max_collisions: int
    score_weight_nav: float
    score_weight_hover: float
    score_weight_wp: float
    nav_time_weight: float
    nav_smooth_weight: float
    phase: torch.Tensor
    progress_buf: torch.Tensor
    nav_steps: torch.Tensor
    hover_steps: torch.Tensor
    arrival_step: torch.Tensor
    hover_timer: torch.Tensor
    action_error_order1: torch.Tensor
    action_error_order2: torch.Tensor
    state_smoothness_cost: torch.Tensor
    smoothness_accum: torch.Tensor
    reward_return: torch.Tensor
    goal_marker_positions: torch.Tensor
    obstacle_positions: torch.Tensor
    obstacle_radii: torch.Tensor
    active_obstacle_mask: torch.Tensor
    active_radius_bucket: torch.Tensor
    num_obstacles: torch.Tensor
    base_obstacle_radius: torch.Tensor
    active_waypoint_mask: torch.Tensor
    num_waypoints: torch.Tensor
    waypoint_visited: torch.Tensor
    waypoint_visited_step: torch.Tensor
    waypoint_positions: torch.Tensor
    collision_count: torch.Tensor
    collision_exceeded: torch.Tensor
    timeout_flag: torch.Tensor
    arrival_success_flag: torch.Tensor
    hover_success_flag: torch.Tensor
    hover_failed_flag: torch.Tensor
    success_flag: torch.Tensor
    prev_collision_flag: torch.Tensor
    waypoint_score_sum: torch.Tensor
    drone_state: torch.Tensor
    prev_action: torch.Tensor
    prev_action_diff: torch.Tensor
    last_applied_action: torch.Tensor
    action_diff: Optional[torch.Tensor]
    action_diff_jerk: Optional[torch.Tensor]
    effort: Optional[torch.Tensor]
    prev_linear_velocity: torch.Tensor
    prev_angular_velocity: torch.Tensor
    prev_distance_to_target: torch.Tensor
    hover_error_sum: torch.Tensor
    hover_error_count: torch.Tensor
    hover_precision: torch.Tensor
    obstacle_clearance: torch.Tensor
    obstacle_collision: torch.Tensor
    out_of_bounds: torch.Tensor

    @classmethod
    def from_env(cls, env: Any) -> "RewardStateSnapshot":
        drone_state = env._latest_drone_state if getattr(env, "_latest_drone_state", None) is not None else env.drone.get_state()
        drone_pos = drone_state[..., :3]
        hover_precision = env._compute_hover_precision()
        obstacle_clearance, obstacle_collision = env._compute_obstacle_metrics(drone_pos)
        out_of_bounds = env._compute_out_of_bounds(drone_pos)
        return cls(
            device=torch.device(env.device),
            num_envs=int(env.num_envs),
            nav_phase=int(env.nav_phase),
            hover_phase=int(env.hover_phase),
            arrival_radius=float(env.arrival_radius),
            stall_progress_threshold=float(env.stall_progress_threshold),
            nav_max_steps=int(env.nav_max_steps),
            hover_max_steps=int(env.hover_max_steps),
            drone_radius=float(env.drone_radius),
            obstacle_safe_distance=float(env.obstacle_safe_distance),
            waypoint_collect_radius=float(env.waypoint_collect_radius),
            max_collisions=int(env.max_collisions),
            score_weight_nav=float(env.score_weight_nav),
            score_weight_hover=float(env.score_weight_hover),
            score_weight_wp=float(env.score_weight_wp),
            nav_time_weight=float(env.nav_time_weight),
            nav_smooth_weight=float(env.nav_smooth_weight),
            phase=_clone_tensor(env.phase),
            progress_buf=_clone_tensor(env.progress_buf),
            nav_steps=_clone_tensor(env.nav_steps),
            hover_steps=_clone_tensor(env.hover_steps),
            arrival_step=_clone_tensor(env.arrival_step),
            hover_timer=_clone_tensor(env.hover_timer),
            action_error_order1=_clone_tensor(env.action_error_order1),
            action_error_order2=_clone_tensor(env.action_error_order2),
            state_smoothness_cost=_clone_tensor(env.state_smoothness_cost),
            smoothness_accum=_clone_tensor(env.smoothness_accum),
            reward_return=_clone_tensor(env.reward_return),
            goal_marker_positions=_clone_tensor(env.goal_marker_positions),
            obstacle_positions=_clone_tensor(env.obstacle_positions),
            obstacle_radii=_clone_tensor(env.obstacle_radii),
            active_obstacle_mask=_clone_tensor(env.active_obstacle_mask),
            active_radius_bucket=_clone_tensor(env.active_radius_bucket),
            num_obstacles=_clone_tensor(env.num_obstacles),
            base_obstacle_radius=_clone_tensor(env.base_obstacle_radius),
            active_waypoint_mask=_clone_tensor(env.active_waypoint_mask),
            num_waypoints=_clone_tensor(env.num_waypoints),
            waypoint_visited=_clone_tensor(env.waypoint_visited),
            waypoint_visited_step=_clone_tensor(env.waypoint_visited_step),
            waypoint_positions=_clone_tensor(env.waypoint_positions),
            collision_count=_clone_tensor(env.collision_count),
            collision_exceeded=_clone_tensor(env.collision_exceeded),
            timeout_flag=_clone_tensor(env.timeout_flag),
            arrival_success_flag=_clone_tensor(env.arrival_success_flag),
            hover_success_flag=_clone_tensor(env.hover_success_flag),
            hover_failed_flag=_clone_tensor(env.hover_failed_flag),
            success_flag=_clone_tensor(env.success_flag),
            prev_collision_flag=_clone_tensor(env.prev_collision_flag),
            waypoint_score_sum=_clone_tensor(env.waypoint_score_sum),
            drone_state=_clone_tensor(drone_state),
            prev_action=_clone_tensor(env.prev_action),
            prev_action_diff=_clone_tensor(env.prev_action_diff),
            last_applied_action=_clone_tensor(env.last_applied_action),
            action_diff=_clone_optional_tensor(getattr(env, "action_diff", None)),
            action_diff_jerk=_clone_optional_tensor(getattr(env, "action_diff_jerk", None)),
            effort=_clone_optional_tensor(getattr(env, "effort", None)),
            prev_linear_velocity=_clone_tensor(env.prev_linear_velocity),
            prev_angular_velocity=_clone_tensor(env.prev_angular_velocity),
            prev_distance_to_target=_clone_tensor(env.prev_distance_to_target),
            hover_error_sum=_clone_tensor(env.hover_error_sum),
            hover_error_count=_clone_tensor(env.hover_error_count),
            hover_precision=_clone_tensor(hover_precision),
            obstacle_clearance=_clone_tensor(obstacle_clearance),
            obstacle_collision=_clone_tensor(obstacle_collision),
            out_of_bounds=_clone_tensor(out_of_bounds),
        )

    def _compute_hover_precision(self):
        return self.hover_precision

    def _compute_obstacle_metrics(self, _drone_pos):
        return self.obstacle_clearance, self.obstacle_collision

    def _compute_out_of_bounds(self, _drone_pos):
        return self.out_of_bounds

__all__ = ["RewardStateSnapshot"]
