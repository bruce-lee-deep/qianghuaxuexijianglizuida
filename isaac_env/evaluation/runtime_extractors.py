#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from .score_utils import safe_float, safe_int, to_serializable


def _get_task_env(drone_env):
    base_env = getattr(drone_env.env, "env", None)
    if base_env is None:
        return None
    return getattr(base_env, "base_env", base_env)




def _get_first_done_payload(drone_env, env_idx=0):
    runtime = getattr(drone_env, "env", None)
    first_done_payloads = getattr(runtime, "eval_first_done_payloads", None)
    if not isinstance(first_done_payloads, list) or env_idx >= len(first_done_payloads):
        return None
    payload = first_done_payloads[env_idx]
    return payload if isinstance(payload, dict) else None


def _normalize_layout_waypoint_steps(layout_dict):
    if not isinstance(layout_dict, dict):
        return {}

    normalized = dict(layout_dict)
    waypoints = normalized.get("waypoints")
    if not isinstance(waypoints, list):
        return normalized

    normalized_waypoints = []
    for waypoint in waypoints:
        if not isinstance(waypoint, dict):
            normalized_waypoints.append(waypoint)
            continue
        normalized_waypoint = dict(waypoint)
        if normalized_waypoint.get("visited_step") is None:
            normalized_waypoint["visited_step"] = -1
        normalized_waypoints.append(normalized_waypoint)
    normalized["waypoints"] = normalized_waypoints
    return normalized


def _build_first_done_snapshot_summary(first_done_payloads, expected_env_count=None):
    if not isinstance(first_done_payloads, list):
        raise RuntimeError("评估环境缺少首局 done 快照缓存: eval_first_done_payloads")

    env_count = len(first_done_payloads)
    if expected_env_count is not None and env_count != int(expected_env_count):
        raise RuntimeError(
            f"评估环境首局 done 快照数量异常: expected={int(expected_env_count)}, actual={env_count}"
        )

    missing_envs = [idx for idx, payload in enumerate(first_done_payloads) if not isinstance(payload, dict)]
    return {
        "env_count": env_count,
        "snapshot_count": env_count - len(missing_envs),
        "missing_envs": missing_envs,
    }


def _get_runtime_first_done_snapshot_summary(drone_env, expected_env_count=None):
    runtime = getattr(drone_env, "env", None)
    first_done_payloads = getattr(runtime, "eval_first_done_payloads", None)
    return _build_first_done_snapshot_summary(first_done_payloads, expected_env_count=expected_env_count)


def validate_eval_first_done_payloads(drone_env, expected_env_count=None):
    if hasattr(drone_env, "apply_function") and callable(getattr(drone_env, "apply_function")):
        summary = drone_env.apply_function(
            "get_eval_first_done_snapshot_summary",
            expected_env_count=expected_env_count,
        )
        if isinstance(summary, dict):
            return summary
        raise RuntimeError(f"评估环境首局 done 快照摘要返回异常: {type(summary).__name__}")

    return _get_runtime_first_done_snapshot_summary(drone_env, expected_env_count=expected_env_count)


def _validate_terminal_score_fields(result_dict):
    if not isinstance(result_dict, dict):
        return {}

    required_score_fields = (
        "nav_score",
        "hover_score",
        "waypoint_score",
        "nav_score_raw",
        "hover_score_raw",
        "wp_score_raw",
    )
    missing_fields = [field_name for field_name in required_score_fields if field_name not in result_dict]
    if missing_fields:
        raise RuntimeError(f"评估结果缺少必需评分字段: {', '.join(missing_fields)}")

    return dict(result_dict)


def build_layout_from_runtime(drone_env, env_idx=0):
    task_env = _get_task_env(drone_env)
    if task_env is None:
        return None

    obstacle_positions = getattr(task_env, "obstacle_positions", None)
    obstacle_radii = getattr(task_env, "obstacle_radii", None)
    num_obstacles = getattr(task_env, "num_obstacles", None)
    base_obstacle_radius = getattr(task_env, "base_obstacle_radius", None)
    waypoint_positions = getattr(task_env, "waypoint_positions", None)
    waypoint_visited = getattr(task_env, "waypoint_visited", None)
    waypoint_visited_step = getattr(task_env, "waypoint_visited_step", None)
    num_waypoints = getattr(task_env, "num_waypoints", None)
    arena_size = getattr(task_env, "arena_size", None)
    start_pos = getattr(task_env, "start_pos", None)
    goal_pos = getattr(task_env, "goal_pos", None)
    drone_radius = getattr(task_env, "drone_radius", None)
    arrival_radius = getattr(task_env, "arrival_radius", None)
    waypoint_collect_radius = getattr(task_env, "waypoint_collect_radius", None)

    if obstacle_positions is None or obstacle_radii is None or num_obstacles is None:
        return None

    env_obstacle_num = int(safe_float(num_obstacles[env_idx], default=0.0))
    positions = []
    obstacles = []
    for obstacle_idx in range(env_obstacle_num):
        x = safe_float(obstacle_positions[env_idx, obstacle_idx, 0])
        y = safe_float(obstacle_positions[env_idx, obstacle_idx, 1])
        radius = safe_float(obstacle_radii[env_idx, obstacle_idx])
        positions.append([x, y])
        obstacles.append({"x": x, "y": y, "radius": radius})

    env_waypoint_num = int(safe_float(num_waypoints[env_idx], default=0.0)) if num_waypoints is not None else 0
    waypoints = []
    effective_waypoint_radius = safe_float(waypoint_collect_radius, default=safe_float(arrival_radius, default=0.0))
    for waypoint_idx in range(env_waypoint_num):
        pos = [
            safe_float(waypoint_positions[env_idx, waypoint_idx, 0]),
            safe_float(waypoint_positions[env_idx, waypoint_idx, 1]),
            safe_float(waypoint_positions[env_idx, waypoint_idx, 2]),
        ] if waypoint_positions is not None else []
        visited = bool(safe_float(waypoint_visited[env_idx, waypoint_idx], default=0.0) >= 0.5) if waypoint_visited is not None else False
        visited_step = int(safe_float(waypoint_visited_step[env_idx, waypoint_idx], default=-1.0)) if waypoint_visited_step is not None else -1
        waypoints.append(
            {
                "id": waypoint_idx,
                "pos": pos,
                "visited": visited,
                "visited_step": visited_step,
                "effective_radius": effective_waypoint_radius,
            }
        )

    return {
        "num_obstacles": env_obstacle_num,
        "radius": safe_float(base_obstacle_radius[env_idx] if base_obstacle_radius is not None else 0.0),
        "positions": positions,
        "space_size": to_serializable(arena_size) if arena_size is not None else [],
        "start_pos": to_serializable(start_pos) if start_pos is not None else [],
        "goal_pos": to_serializable(goal_pos) if goal_pos is not None else [],
        "obstacles": obstacles,
        "num_waypoints": env_waypoint_num,
        "waypoints": waypoints,
        "drone_radius": safe_float(drone_radius, default=0.0),
        "arrival_radius": safe_float(arrival_radius, default=0.0),
        "goal_radius": safe_float(arrival_radius, default=0.0),
        "waypoint_radius": safe_float(waypoint_collect_radius, default=safe_float(arrival_radius, default=0.0)),
    }


def build_trajectory_from_runtime(drone_env, env_idx=0):
    task_env = _get_task_env(drone_env)
    if task_env is None:
        return []

    trajectory_buffer = getattr(task_env, "trajectory_buffer", None)
    if trajectory_buffer is None or env_idx >= len(trajectory_buffer):
        return []

    return to_serializable(trajectory_buffer[env_idx])


def build_result_from_runtime(drone_env, env_idx=0):
    task_env = _get_task_env(drone_env)
    if task_env is None:
        return None

    success_flag = getattr(task_env, "success_flag", None)
    timeout_flag = getattr(task_env, "timeout_flag", None)
    collision_exceeded = getattr(task_env, "collision_exceeded", None)
    collision_count = getattr(task_env, "collision_count", None)
    arrival_step = getattr(task_env, "arrival_step", None)
    hover_steps = getattr(task_env, "hover_steps", None)
    nav_steps = getattr(task_env, "nav_steps", None)
    stats = getattr(task_env, "stats", None)
    progress_buf = getattr(task_env, "progress_buf", None)
    max_collisions = getattr(task_env, "max_collisions", None)
    nav_max_steps = getattr(task_env, "nav_max_steps", None)
    hover_max_steps = getattr(task_env, "hover_max_steps", None)
    dt = getattr(task_env, "dt", None)

    if success_flag is None or timeout_flag is None or collision_exceeded is None:
        return None

    success = bool(safe_float(success_flag[env_idx]) >= 0.5)
    timeout = bool(safe_float(timeout_flag[env_idx]) >= 0.5)
    collision_failed = bool(safe_float(collision_exceeded[env_idx]) >= 0.5)
    if success:
        status = "success"
    elif timeout:
        status = "timeout"
    else:
        status = "failed"

    def _stat_value(key, default=0.0):
        if stats is None or key not in stats.keys():
            return default
        return safe_float(stats[key][env_idx], default=default)

    return {
        "status": status,
        "score": _stat_value("total_score"),
        "total_score": _stat_value("total_score"),
        "nav_score": _stat_value("nav_score"),
        "hover_score": _stat_value("hover_score"),
        "nav_coeff": _stat_value("nav_coeff"),
        "hover_coeff": _stat_value("hover_coeff"),
        "nav_score_raw": _stat_value("nav_score_raw"),
        "hover_score_raw": _stat_value("hover_score_raw"),
        "wp_score_raw": _stat_value("wp_score_raw"),
        "time_norm": _stat_value("time_norm"),
        "smooth_norm": _stat_value("smooth_norm"),
        "arrival_step": int(safe_float(arrival_step[env_idx], default=-1.0)) if arrival_step is not None else -1,
        "episode_len": int(safe_float(progress_buf[env_idx], default=0.0)) if progress_buf is not None else 0,
        "nav_steps": int(safe_float(nav_steps[env_idx], default=0.0)) if nav_steps is not None else 0,
        "hover_steps": int(safe_float(hover_steps[env_idx], default=0.0)) if hover_steps is not None else 0,
        "collision_count": int(safe_float(collision_count[env_idx], default=0.0)) if collision_count is not None else 0,
        "collision_rate": _stat_value("collision_rate"),
        "hover_precision": _stat_value("hover_precision"),
        "success": success,
        "timeout": timeout,
        "failed": (not success) and (not timeout),
        "collision_exceeded": collision_failed,
        "max_collisions": int(max_collisions) if max_collisions is not None else 0,
        "nav_max_steps": int(nav_max_steps) if nav_max_steps is not None else 0,
        "hover_max_steps": int(hover_max_steps) if hover_max_steps is not None else 0,
        "sim_dt": float(dt) if dt is not None else 0.0,
        "waypoint_score": _stat_value("waypoint_score"),
        "waypoints_visited": _stat_value("waypoints_visited"),
        "waypoints_total": _stat_value("waypoints_total"),
        "remaining_waypoints": _stat_value(
            "remaining_waypoints",
            max(
                _stat_value("waypoints_total") - _stat_value("waypoints_visited"),
                0.0,
            ),
        ),
        "visited_waypoints": _stat_value("waypoints_visited"),
        "total_waypoints": _stat_value("waypoints_total"),
        "waypoint_score_sum": _stat_value("waypoint_score_sum", _stat_value("waypoints_visited")),
        "arrival_success": _stat_value("arrival_success"),
        "hover_success": _stat_value("hover_success"),
        "hover_failed": _stat_value("hover_failed"),
    }


def build_end_info(drone_env, env_idx=0):
    task_env = _get_task_env(drone_env)
    first_done_payload = _get_first_done_payload(drone_env, env_idx=env_idx)
    payload = first_done_payload
    if payload is None and task_env is not None:
        latest_episode_payload = getattr(task_env, "latest_episode_payload", None)
        if latest_episode_payload is not None and env_idx < len(latest_episode_payload):
            payload = latest_episode_payload[env_idx]

    if payload is None:
        payload = {}
    payload = to_serializable(payload)
    if not isinstance(payload, dict):
        payload = {}

    payload_layout = _normalize_layout_waypoint_steps(payload.get("layout") if isinstance(payload.get("layout"), dict) else {})
    payload_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}

    if first_done_payload is not None:
        layout = payload_layout or {}
        result = _validate_terminal_score_fields(payload_result or {})
    else:
        runtime_layout = build_layout_from_runtime(drone_env, env_idx) or {}
        layout = {**runtime_layout, **payload_layout}

        runtime_result = build_result_from_runtime(drone_env, env_idx) or {}
        result = _validate_terminal_score_fields({**runtime_result, **payload_result})

    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list):
        trajectory = [] if first_done_payload is not None else build_trajectory_from_runtime(drone_env, env_idx)
    trajectory = trajectory or []

    return {
        "protocol_version": "v1",
        "project_code": "drone_obstacle_nav",
        "game_id": drone_env.game_id,
        "layout": layout,
        "trajectory": trajectory,
        "result": result,
    }


def build_all_end_info_dicts(drone_env):
    task_env = _get_task_env(drone_env)
    env_count = 0
    if task_env is not None:
        latest_episode_payload = getattr(task_env, "latest_episode_payload", None)
        if isinstance(latest_episode_payload, list):
            env_count = len(latest_episode_payload)

    runtime = getattr(drone_env, "env", None)
    first_done_payloads = getattr(runtime, "eval_first_done_payloads", None)
    if isinstance(first_done_payloads, list):
        env_count = max(env_count, len(first_done_payloads))

    if env_count <= 0:
        env_count = max(safe_int(getattr(drone_env, "env_nums", 0), 0), 1)

    end_info_dicts = []
    for env_idx in range(env_count):
        end_info_dict = build_end_info(drone_env, env_idx=env_idx)
        if not isinstance(end_info_dict, dict):
            end_info_dict = {}
        end_info_dicts.append(end_info_dict)
    return end_info_dicts
