#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Per-obstacle-count control strategies.

Each strategy encapsulates the behavioral differences (not just parameter
scaling) for a specific obstacle density.  The RuleMPCController selects
the appropriate strategy at runtime based on the observed obstacle count.

Strategies differ in:
  - Waypoint selection aggressiveness (direct-goal vs via-waypoints)
  - Speed profile shape (max speed, accel/decel fractions)
  - Danger / clearance thresholds
  - Obstacle avoidance reactivity
  - Hard-safety trigger distances
  - Speed-brake parameters
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ObstacleStrategyParams:
    """All tunable parameters for one obstacle-count strategy."""

    # ---- Waypoint selection ----
    # Higher = more willing to skip waypoints and go direct to goal.
    direct_goal_bias: float = 0.0
    # Higher = more willing to pick a waypoint even if segment is tight.
    waypoint_risk_tolerance: float = 1.0

    # ---- Speed profile (_build_reference_velocity) ----
    wp_speed_base: float = 0.45       # m/s, base cruise speed
    wp_speed_gain: float = 0.42       # m/s per m of segment length
    wp_speed_cap: float = 1.35        # m/s, absolute speed ceiling
    wp_min_speed: float = 0.28        # m/s, minimum cruise speed
    wp_accel_frac: float = 0.25       # fraction of segment for acceleration
    wp_decel_frac: float = 0.25       # fraction of segment for deceleration
    wp_finish_speed_frac: float = 0.32  # speed at finish as fraction of max
    safe_segment_speed_scale: float = 1.0  # multiplier for safe segments
    goal_fast_speed_scale: float = 1.25    # speed scale when far from goal
    goal_brake_speed_scale: float = 0.35   # speed scale when braking to goal

    # ---- Danger detection ----
    danger_wp_clearance: float = 0.25     # m, waypoint-to-obstacle clearance
    danger_segment_clearance: float = 0.35  # m, segment-to-obstacle clearance
    critical_wp_clearance: float = 0.05    # m, critical (never pick)
    hard_segment_clearance: float = 0.08   # m, hard penalty threshold

    # ---- Obstacle avoidance (_build_obstacle_avoidance_target) ----
    obs_base_margin: float = 0.26       # m, base margin around obstacles
    obs_speed_margin_gain: float = 0.08  # m per m/s of speed
    obs_detour_margin: float = 0.42     # m, detour margin
    obs_detour_advance: float = 0.45    # m, look-ahead for detour
    obs_wall_margin: float = 0.32       # m, margin from arena walls
    obs_normal_trigger_min: float = 0.60  # m, trigger dist for normal avoidance
    obs_lookahead_max: float = 1.00     # m, max lookahead for obstacles
    obs_avoid_hold_frames: int = 18     # frames to hold avoidance

    # ---- Emergency obstacle safety (_apply_obstacle_safety_accel) ----
    emergency_margin: float = 0.24      # m
    emergency_gain: float = 2.2
    emergency_accel_max: float = 1.8

    # ---- Hard safety (_apply_hard_obstacle_safety) ----
    hard_safety_clearance: float = 0.18      # m, trigger
    hard_safety_release_clearance: float = 0.26  # m, release
    hard_safety_away_accel: float = 2.4
    hard_safety_brake_gain: float = 1.8
    hard_safety_max_accel: float = 2.6

    # ---- Speed-dependent brake (_apply_speed_dependent_target_brake) ----
    brake_decel: float = 1.8            # m/s^2, brake deceleration
    brake_buffer: float = 0.18          # m, buffer distance
    brake_dist_min: float = 0.35        # m, minimum brake distance
    brake_gain: float = 1.35            # proportional gain
    brake_max_accel: float = 1.8        # m/s^2, max brake accel
    brake_near_obs_clearance: float = 0.65  # m, "near obstacle" threshold
    brake_near_obs_speed: float = 0.25      # m/s, allowed speed near obstacle
    brake_safe_speed: float = 0.55          # m/s, allowed speed in clear space

    # ---- Detour scoring (_score_detour_candidate) ----
    detour_clearance: float = 0.35      # m, clearance limit for detour
    detour_extra_margin: float = 0.10   # m, extra margin for chosen obstacle

    # ---- Post-danger guard ----
    post_danger_guard_frames: int = 45
    post_danger_speed_scale: float = 0.45

    # ---- Waypoint approach ----
    danger_wp_approach_speed_scale: float = 0.75
    danger_wp_brake_dist_frac: float = 0.33


# =========================================================================
# Pre-defined strategies per obstacle count
# =========================================================================
# Design principles for 5 obstacles:
#   - Fewer obstacles = more open corridors = can fly faster AND more direct.
#   - Don't over-reduce safety margins: the natural space advantage is enough.
#   - Braking should still be effective at the higher speeds.
#   - Waypoint selection can favour direct-goal more often.
#   - Obstacle avoidance can be less reactive (fewer things to avoid).

STRATEGY_5 = ObstacleStrategyParams(
    # --- Waypoint selection: more direct ---
    direct_goal_bias=0.12,
    waypoint_risk_tolerance=1.20,

    # --- Speed: faster but capped for safety ---
    # Critical: wp_speed_cap * safe_segment_speed_scale must stay <= 1.65
    # and wp_speed_cap * goal_fast_speed_scale must stay <= 1.75 to avoid
    # loss of control at high speeds (seen at 2.17 m/s).
    wp_speed_base=0.50,          # +11% vs default 0.45
    wp_speed_gain=0.46,          # +10% vs default 0.42
    wp_speed_cap=1.45,           # +7% vs default 1.35 (was 1.55)
    wp_min_speed=0.30,           # +7% vs default 0.28
    wp_accel_frac=0.22,          # quicker accel
    wp_decel_frac=0.28,          # slightly longer decel for safety
    wp_finish_speed_frac=0.30,   # slightly lower finish speed
    safe_segment_speed_scale=1.08,   # 1.45*1.08=1.57 safe (was 1.15→1.78)
    goal_fast_speed_scale=1.18,      # 1.45*1.18=1.71 fast (was 1.30→2.02)
    goal_brake_speed_scale=0.38,     # less aggressive brake-to-goal

    # --- Danger detection: slightly relaxed ---
    danger_wp_clearance=0.22,        # -12%
    danger_segment_clearance=0.30,   # -14%
    critical_wp_clearance=0.05,      # unchanged (safety floor)
    hard_segment_clearance=0.07,     # -12%

    # --- Obstacle avoidance: less defensive ---
    obs_base_margin=0.24,            # -8% (was 0.22, restored for safety)
    obs_speed_margin_gain=0.06,      # -25%
    obs_detour_margin=0.35,          # -17%
    obs_detour_advance=0.38,         # -16%
    obs_wall_margin=0.28,            # -12%
    obs_normal_trigger_min=0.50,     # trigger later
    obs_lookahead_max=0.90,          # look less far ahead
    obs_avoid_hold_frames=14,        # release avoidance faster

    # --- Emergency safety: slightly relaxed but still safe ---
    emergency_margin=0.22,           # -8% (was 0.20, restored for safety)
    emergency_gain=2.0,              # -9%
    emergency_accel_max=1.6,         # -11%

    # --- Hard safety: still present but less trigger-happy ---
    hard_safety_clearance=0.15,          # -17%
    hard_safety_release_clearance=0.22,  # -15%
    hard_safety_away_accel=2.2,          # -8%
    hard_safety_brake_gain=1.6,          # -11%
    hard_safety_max_accel=2.3,           # -12%

    # --- Braking: tuned for higher cruise speed ---
    brake_decel=1.9,                 # slightly stronger decel
    brake_buffer=0.20,               # slightly larger buffer (higher speeds)
    brake_dist_min=0.38,             # slightly longer min dist
    brake_gain=1.30,                 # slightly less aggressive
    brake_max_accel=1.7,             # slightly lower max
    brake_near_obs_clearance=0.55,   # -15% (tighter "near" definition)
    brake_near_obs_speed=0.30,       # +20% (can go faster near obstacles)
    brake_safe_speed=0.65,           # +18% (higher safe speed)

    # --- Detour: smaller detours ---
    detour_clearance=0.30,           # -14%
    detour_extra_margin=0.08,        # -20%

    # --- Post-danger: recover faster ---
    post_danger_guard_frames=35,     # -22% (shorter guard)
    post_danger_speed_scale=0.52,    # +16% (faster after danger)

    # --- Waypoint approach ---
    danger_wp_approach_speed_scale=0.82,   # +9% (faster approach to danger wp)
    danger_wp_brake_dist_frac=0.30,        # -9% (brake slightly later)
)


# Default strategy (8 obstacles — most cautious, matches original defaults)
STRATEGY_DEFAULT = ObstacleStrategyParams()


# Registry: obstacle_count -> strategy
# Obstacle counts 4, 5, 6 all use STRATEGY_5 (most validated params).
# Obstacle counts 7, 8 fall back to STRATEGY_DEFAULT (reserved for agent_ppo_78).
STRATEGY_REGISTRY = {
    4: STRATEGY_5,
    5: STRATEGY_5,
    6: STRATEGY_5,
}


def get_strategy(obs_count: int) -> ObstacleStrategyParams:
    """Return the strategy for a given obstacle count.

    Falls back to STRATEGY_DEFAULT for unknown counts.
    """
    return STRATEGY_REGISTRY.get(obs_count, STRATEGY_DEFAULT)


__all__ = [
    "ObstacleStrategyParams",
    "STRATEGY_5",
    "STRATEGY_DEFAULT",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
