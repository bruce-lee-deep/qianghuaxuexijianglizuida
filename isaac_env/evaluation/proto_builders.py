#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import json

from google.protobuf.json_format import MessageToJson
from arena_proto.drone_obstacle_nav.custom_pb2 import EndInfo, Frames, Layout, Obstacle, Position2D, StartInfo, TrajectoryFile, TrajectoryPoint, Waypoint

from .score_utils import ensure_float_list, safe_float, safe_int, safe_mean


REQUIRED_RESULT_SCORE_FIELDS = (
    "nav_score",
    "hover_score",
    "waypoint_score",
    "nav_score_raw",
    "hover_score_raw",
    "wp_score_raw",
)


END_INFO_DECIMAL_PLACES = 2
END_INFO_NUMERIC_FIELDS = (
    "score",
    "total_score",
    "nav_score",
    "hover_score",
    "nav_coeff",
    "hover_coeff",
    "nav_score_raw",
    "hover_score_raw",
    "wp_score_raw",
    "time_norm",
    "smooth_norm",
    "remaining_waypoints",
    "arrival_success",
    "hover_success",
    "hover_failed",
    "arrival_step",
    "episode_len",
    "nav_steps",
    "hover_steps",
    "collision_count",
    "collision_rate",
    "hover_precision",
    "success",
    "timeout",
    "failed",
    "collision_exceeded",
    "max_collisions",
    "nav_max_steps",
    "hover_max_steps",
    "sim_dt",
    "game_count",
    "obstacle_radius",
    "obstacle_num",
    "waypoint_score",
    "waypoints_visited",
    "waypoints_total",
    "visited_waypoints",
    "total_waypoints",
    "drone_radius",
    "arrival_radius",
    "goal_radius",
    "waypoint_radius",
    "score_sum",
    "total_score_sum",
    "nav_score_sum",
    "hover_score_sum",
    "arrival_step_sum",
    "episode_len_sum",
    "nav_steps_sum",
    "hover_steps_sum",
    "collision_count_sum",
    "collision_rate_sum",
    "hover_precision_sum",
    "collision_exceeded_sum",
    "max_collisions_sum",
    "nav_max_steps_sum",
    "hover_max_steps_sum",
    "sim_dt_sum",
    "waypoints_visited_sum",
    "waypoints_total_sum",
    "waypoint_score_sum",
    "nav_coeff_sum",
    "hover_coeff_sum",
    "nav_score_raw_sum",
    "hover_score_raw_sum",
    "wp_score_raw_sum",
    "time_norm_sum",
    "smooth_norm_sum",
    "visited_waypoints_sum",
    "total_waypoints_sum",
    "remaining_waypoints_sum",
    "arrival_success_sum",
    "hover_success_sum",
    "hover_failed_sum",
    "drone_radius_sum",
    "arrival_radius_sum",
    "goal_radius_sum",
    "waypoint_radius_sum",
    "obstacle_radius_sum",
    "obstacle_num_sum",
)


EXPORT_SCORE_DETAIL_SUM_FIELDS = tuple(field_name for field_name in END_INFO_NUMERIC_FIELDS if field_name.endswith("_sum"))


AGGREGATE_SCORE_FIELDS = (
    "score",
    "total_score",
    "nav_score",
    "hover_score",
    "nav_coeff",
    "hover_coeff",
    "nav_score_raw",
    "hover_score_raw",
    "wp_score_raw",
    "time_norm",
    "smooth_norm",
    "remaining_waypoints",
    "arrival_success",
    "hover_success",
    "hover_failed",
    "arrival_step",
    "episode_len",
    "nav_steps",
    "hover_steps",
    "collision_count",
    "collision_rate",
    "hover_precision",
    "success",
    "timeout",
    "failed",
    "collision_exceeded",
    "max_collisions",
    "nav_max_steps",
    "hover_max_steps",
    "sim_dt",
    "obstacle_radius",
    "obstacle_num",
    "waypoint_score",
    "waypoints_visited",
    "waypoints_total",
    "visited_waypoints",
    "total_waypoints",
    "drone_radius",
    "arrival_radius",
    "goal_radius",
    "waypoint_radius",
)


AGGREGATE_SUM_FIELDS = (
    "success",
    "timeout",
    "failed",
)


AGGREGATE_MEAN_FIELDS = tuple(field_name for field_name in AGGREGATE_SCORE_FIELDS if field_name not in AGGREGATE_SUM_FIELDS)
AGGREGATE_SUM_FIELD_MAP = {field_name: f"{field_name}_sum" for field_name in AGGREGATE_MEAN_FIELDS}


def _format_task_type(obstacle_num, obstacle_radius, waypoint_count):
    radius_code = safe_int(round(safe_float(obstacle_radius, 0.0) * 100.0), 0)
    return f"obs_{safe_int(obstacle_num, 0)}_r_{radius_code}_wp_{safe_int(waypoint_count, 0)}"


def build_layout_message(layout_dict):
    if not isinstance(layout_dict, dict):
        layout_dict = {}
    layout_message = Layout(
        num_obstacles=safe_int(layout_dict.get("num_obstacles"), 0),
        radius=safe_float(layout_dict.get("radius"), 0.0),
    )
    layout_message.space_size.extend(ensure_float_list(layout_dict.get("space_size")))
    layout_message.start_pos.extend(ensure_float_list(layout_dict.get("start_pos")))
    layout_message.goal_pos.extend(ensure_float_list(layout_dict.get("goal_pos")))

    for position in layout_dict.get("positions", []) or []:
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        layout_message.positions.append(Position2D(x=safe_float(position[0], 0.0), y=safe_float(position[1], 0.0)))

    for obstacle in layout_dict.get("obstacles", []) or []:
        if not isinstance(obstacle, dict):
            continue
        layout_message.obstacles.append(
            Obstacle(
                x=safe_float(obstacle.get("x"), 0.0),
                y=safe_float(obstacle.get("y"), 0.0),
                radius=safe_float(obstacle.get("radius"), 0.0),
            )
        )

    layout_message.num_waypoints = safe_int(layout_dict.get("num_waypoints"), 0)
    for waypoint in layout_dict.get("waypoints", []) or []:
        if not isinstance(waypoint, dict):
            continue
        visited_step = safe_int(waypoint.get("visited_step"), -1)
        layout_message.waypoints.append(
            Waypoint(
                id=safe_int(waypoint.get("id"), 0),
                pos=ensure_float_list(waypoint.get("pos")),
                visited=bool(waypoint.get("visited", False)),
                visited_step=visited_step,
                effective_radius=safe_float(waypoint.get("effective_radius"), 0.0),
            )
        )

    layout_message.drone_radius = safe_float(layout_dict.get("drone_radius"), 0.0)
    layout_message.arrival_radius = safe_float(layout_dict.get("arrival_radius"), 0.0)
    layout_message.goal_radius = safe_float(layout_dict.get("goal_radius", layout_dict.get("arrival_radius")), 0.0)
    layout_message.waypoint_radius = safe_float(layout_dict.get("waypoint_radius", layout_dict.get("arrival_radius")), 0.0)

    return layout_message


def build_trajectory_messages(trajectory_list):
    trajectory_messages = []
    for item in trajectory_list or []:
        if not isinstance(item, dict):
            continue
        trajectory_message = TrajectoryPoint(step=safe_int(item.get("step"), 0), timestamp=safe_float(item.get("timestamp"), 0.0))
        trajectory_message.pos.extend(ensure_float_list(item.get("pos")))
        trajectory_message.vel.extend(ensure_float_list(item.get("vel")))
        trajectory_message.action.extend(ensure_float_list(item.get("action")))
        trajectory_message.quat.extend(ensure_float_list(item.get("quat")))
        trajectory_message.rotation.extend(ensure_float_list(item.get("rotation")))
        trajectory_messages.append(trajectory_message)
    return trajectory_messages


def build_start_info_message(game_id):
    return StartInfo(protocol_version="v1", project_code="drone_obstacle_nav", game_id=str(game_id))


def build_trajectory_file_message(env_idx, trajectory_list, layout_dict=None):
    trajectory_file_message = TrajectoryFile(protocol_version="v1", env_index=safe_int(env_idx, 0))
    if layout_dict is not None:
        trajectory_file_message.layout.CopyFrom(build_layout_message(layout_dict))
    trajectory_file_message.trajectory.extend(build_trajectory_messages(trajectory_list))
    return trajectory_file_message


def build_frames_message():
    return Frames(protocol_version="v1")


def _extract_layout_dict(end_info_dict):
    layout_dict = end_info_dict.get("layout", {}) if isinstance(end_info_dict, dict) else {}
    return layout_dict if isinstance(layout_dict, dict) else {}


def _extract_result_dict(end_info_dict):
    result_dict = end_info_dict.get("result", {}) if isinstance(end_info_dict, dict) else {}
    return result_dict if isinstance(result_dict, dict) else {}


def _require_result_score_fields(result_dict, env_idx):
    missing_fields = [field_name for field_name in REQUIRED_RESULT_SCORE_FIELDS if field_name not in result_dict]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise RuntimeError(f"env_index={safe_int(env_idx, 0)} result 缺少必需评分字段: {missing}")


def _extract_obstacle_radius(layout_dict):
    radius = safe_float(layout_dict.get("radius"), 0.0)
    if radius > 0:
        return radius
    obstacle_radii = [
        safe_float(obstacle.get("radius"), 0.0)
        for obstacle in layout_dict.get("obstacles", []) or []
        if isinstance(obstacle, dict)
    ]
    return safe_mean(obstacle_radii)


def _round_end_info_value(value):
    return round(safe_float(value, 0.0), END_INFO_DECIMAL_PLACES)


def _normalize_end_info_values(values):
    normalized = dict(values)
    for field_name in END_INFO_NUMERIC_FIELDS:
        if field_name in normalized:
            normalized[field_name] = _round_end_info_value(normalized[field_name])
    return normalized


def _build_score_detail_values(env_idx, end_info_dict):
    layout_dict = _extract_layout_dict(end_info_dict)
    result_dict = _extract_result_dict(end_info_dict)
    _require_result_score_fields(result_dict, env_idx)
    obstacle_radius = _extract_obstacle_radius(layout_dict)
    obstacle_num = safe_float(layout_dict.get("num_obstacles"), 0.0)
    waypoint_count = safe_float(layout_dict.get("num_waypoints"), result_dict.get("total_waypoints", result_dict.get("waypoints_total", 0.0)))
    values = {
        "env_index": safe_int(env_idx, 0),
        "game_count": 1.0,
        "obstacle_radius": obstacle_radius,
        "obstacle_num": obstacle_num,
        "task_type": _format_task_type(obstacle_num, obstacle_radius, waypoint_count),
        "drone_radius": safe_float(layout_dict.get("drone_radius"), 0.0),
        "arrival_radius": safe_float(layout_dict.get("arrival_radius"), 0.0),
        "goal_radius": safe_float(layout_dict.get("goal_radius", layout_dict.get("arrival_radius")), 0.0),
        "waypoint_radius": safe_float(layout_dict.get("waypoint_radius", layout_dict.get("arrival_radius")), 0.0),
        "status": str(result_dict.get("status", "unknown")),
        "score": safe_float(result_dict.get("score"), 0.0),
        "total_score": safe_float(result_dict.get("total_score"), 0.0),
        "nav_score": safe_float(result_dict.get("nav_score"), 0.0),
        "hover_score": safe_float(result_dict.get("hover_score"), 0.0),
        "nav_coeff": safe_float(result_dict.get("nav_coeff"), 0.0),
        "hover_coeff": safe_float(result_dict.get("hover_coeff"), 0.0),
        "nav_score_raw": safe_float(result_dict.get("nav_score_raw"), 0.0),
        "hover_score_raw": safe_float(result_dict.get("hover_score_raw"), 0.0),
        "wp_score_raw": safe_float(result_dict.get("wp_score_raw"), 0.0),
        "time_norm": safe_float(result_dict.get("time_norm"), 0.0),
        "smooth_norm": safe_float(result_dict.get("smooth_norm"), 0.0),
        "remaining_waypoints": safe_float(
            result_dict.get(
                "remaining_waypoints",
                safe_float(result_dict.get("total_waypoints", result_dict.get("waypoints_total", 0.0)), 0.0)
                - safe_float(result_dict.get("visited_waypoints", result_dict.get("waypoints_visited", 0.0)), 0.0),
            ),
            0.0,
        ),
        "arrival_success": 1.0 if bool(result_dict.get("arrival_success", False)) else 0.0,
        "hover_success": 1.0 if bool(result_dict.get("hover_success", False)) else 0.0,
        "hover_failed": 1.0 if bool(result_dict.get("hover_failed", False)) else 0.0,
        "arrival_step": safe_float(result_dict.get("arrival_step"), -1.0),
        "episode_len": safe_float(result_dict.get("episode_len"), 0.0),
        "nav_steps": safe_float(result_dict.get("nav_steps"), 0.0),
        "hover_steps": safe_float(result_dict.get("hover_steps"), 0.0),
        "collision_count": safe_float(result_dict.get("collision_count"), 0.0),
        "collision_rate": safe_float(result_dict.get("collision_rate"), 0.0),
        "hover_precision": safe_float(result_dict.get("hover_precision"), 0.0),
        "success": 1.0 if bool(result_dict.get("success", False)) else 0.0,
        "timeout": 1.0 if bool(result_dict.get("timeout", False)) else 0.0,
        "failed": 1.0 if bool(result_dict.get("failed", False)) else 0.0,
        "collision_exceeded": 1.0 if bool(result_dict.get("collision_exceeded", False)) else 0.0,
        "max_collisions": safe_float(result_dict.get("max_collisions"), 0.0),
        "nav_max_steps": safe_float(result_dict.get("nav_max_steps"), 0.0),
        "hover_max_steps": safe_float(result_dict.get("hover_max_steps"), 0.0),
        "sim_dt": safe_float(result_dict.get("sim_dt"), 0.0),
        "waypoint_score": safe_float(result_dict.get("waypoint_score"), 0.0),
        "waypoints_visited": safe_float(result_dict.get("waypoints_visited"), 0.0),
        "waypoints_total": safe_float(result_dict.get("waypoints_total"), 0.0),
        "visited_waypoints": safe_float(result_dict.get("visited_waypoints", result_dict.get("waypoints_visited")), 0.0),
        "total_waypoints": safe_float(result_dict.get("total_waypoints", result_dict.get("waypoints_total")), 0.0),
        "waypoint_score_sum": safe_float(result_dict.get("waypoint_score_sum"), 0.0),
    }
    for field_name, sum_field_name in AGGREGATE_SUM_FIELD_MAP.items():
        values[sum_field_name] = values[field_name]
    return values


def _summarize_status(score_detail_values):
    statuses = {detail.get("status", "unknown") for detail in score_detail_values if detail.get("status")}
    if not statuses:
        return "unknown"
    if len(statuses) == 1:
        return statuses.pop()
    return "mixed"


def _build_aggregate_values(score_detail_values, task_type=None):
    aggregate_values = {
        "status": _summarize_status(score_detail_values),
        "task_type": task_type if task_type is not None else (
            "mixed"
            if len({detail["task_type"] for detail in score_detail_values if detail.get("task_type")}) != 1
            else (score_detail_values[0].get("task_type", "") if score_detail_values else "")
        ),
        "game_count": float(len(score_detail_values)),
    }
    for field_name in AGGREGATE_SCORE_FIELDS:
        values = [detail[field_name] for detail in score_detail_values]
        if field_name in AGGREGATE_SUM_FIELDS:
            aggregate_values[field_name] = sum(values)
            continue
        aggregate_values[field_name] = safe_mean(values)
        aggregate_values[AGGREGATE_SUM_FIELD_MAP[field_name]] = sum(values)
    return aggregate_values


def _group_score_detail_values(score_detail_values):
    grouped_values = {}
    for detail_values in score_detail_values:
        task_type = detail_values.get("task_type", "")
        grouped_values.setdefault(task_type, []).append(detail_values)

    grouped_score_detail_values = []
    for task_type, grouped_detail_values in sorted(grouped_values.items(), key=lambda item: item[0]):
        grouped_score_detail_values.append(_build_aggregate_values(grouped_detail_values, task_type=task_type))
    return grouped_score_detail_values


def _apply_score_values(message, values):
    values = _normalize_end_info_values(values)
    message_field_names = message.DESCRIPTOR.fields_by_name

    def _set_if_supported(field_name, value):
        if field_name in message_field_names:
            setattr(message, field_name, value)

    message.task_type = values.get("task_type", "")
    message.status = values.get("status", "unknown")
    message.score = values.get("score", 0.0)
    message.total_score = values.get("total_score", 0.0)
    message.nav_score = values.get("nav_score", 0.0)
    message.hover_score = values.get("hover_score", 0.0)
    message.nav_coeff = values.get("nav_coeff", 0.0)
    message.hover_coeff = values.get("hover_coeff", 0.0)
    message.nav_score_raw = values.get("nav_score_raw", 0.0)
    message.hover_score_raw = values.get("hover_score_raw", 0.0)
    message.wp_score_raw = values.get("wp_score_raw", 0.0)
    message.time_norm = values.get("time_norm", 0.0)
    message.smooth_norm = values.get("smooth_norm", 0.0)
    message.remaining_waypoints = values.get("remaining_waypoints", 0.0)
    message.arrival_success = values.get("arrival_success", 0.0)
    message.hover_success = values.get("hover_success", 0.0)
    message.hover_failed = values.get("hover_failed", 0.0)
    message.arrival_step = values.get("arrival_step", 0.0)
    message.episode_len = values.get("episode_len", 0.0)
    message.nav_steps = values.get("nav_steps", 0.0)
    message.hover_steps = values.get("hover_steps", 0.0)
    message.collision_count = values.get("collision_count", 0.0)
    message.collision_rate = values.get("collision_rate", 0.0)
    message.hover_precision = values.get("hover_precision", 0.0)
    message.success = values.get("success", 0.0)
    message.timeout = values.get("timeout", 0.0)
    message.failed = values.get("failed", 0.0)
    message.collision_exceeded = values.get("collision_exceeded", 0.0)
    message.max_collisions = values.get("max_collisions", 0.0)
    message.nav_max_steps = values.get("nav_max_steps", 0.0)
    message.hover_max_steps = values.get("hover_max_steps", 0.0)
    message.sim_dt = values.get("sim_dt", 0.0)
    message.game_count = values.get("game_count", 0.0)
    message.obstacle_radius = values.get("obstacle_radius", 0.0)
    message.obstacle_num = values.get("obstacle_num", 0.0)
    message.waypoint_score = values.get("waypoint_score", 0.0)
    message.waypoints_visited = values.get("waypoints_visited", 0.0)
    message.waypoints_total = values.get("waypoints_total", 0.0)
    message.visited_waypoints = values.get("visited_waypoints", values.get("waypoints_visited", 0.0))
    message.total_waypoints = values.get("total_waypoints", values.get("waypoints_total", 0.0))
    message.drone_radius = values.get("drone_radius", 0.0)
    message.arrival_radius = values.get("arrival_radius", 0.0)
    message.goal_radius = values.get("goal_radius", values.get("arrival_radius", 0.0))
    message.waypoint_radius = values.get("waypoint_radius", values.get("arrival_radius", 0.0))
    _set_if_supported("score_sum", values.get("score_sum", 0.0))
    _set_if_supported("total_score_sum", values.get("total_score_sum", 0.0))
    _set_if_supported("nav_score_sum", values.get("nav_score_sum", 0.0))
    _set_if_supported("hover_score_sum", values.get("hover_score_sum", 0.0))
    _set_if_supported("arrival_step_sum", values.get("arrival_step_sum", 0.0))
    _set_if_supported("episode_len_sum", values.get("episode_len_sum", 0.0))
    _set_if_supported("nav_steps_sum", values.get("nav_steps_sum", 0.0))
    _set_if_supported("hover_steps_sum", values.get("hover_steps_sum", 0.0))
    _set_if_supported("collision_count_sum", values.get("collision_count_sum", 0.0))
    _set_if_supported("collision_rate_sum", values.get("collision_rate_sum", 0.0))
    _set_if_supported("hover_precision_sum", values.get("hover_precision_sum", 0.0))
    _set_if_supported("collision_exceeded_sum", values.get("collision_exceeded_sum", 0.0))
    _set_if_supported("max_collisions_sum", values.get("max_collisions_sum", 0.0))
    _set_if_supported("nav_max_steps_sum", values.get("nav_max_steps_sum", 0.0))
    _set_if_supported("hover_max_steps_sum", values.get("hover_max_steps_sum", 0.0))
    _set_if_supported("sim_dt_sum", values.get("sim_dt_sum", 0.0))
    _set_if_supported("waypoint_score_sum", values.get("waypoint_score_sum", 0.0))
    _set_if_supported("waypoints_visited_sum", values.get("waypoints_visited_sum", 0.0))
    _set_if_supported("waypoints_total_sum", values.get("waypoints_total_sum", 0.0))
    _set_if_supported("nav_coeff_sum", values.get("nav_coeff_sum", 0.0))
    _set_if_supported("hover_coeff_sum", values.get("hover_coeff_sum", 0.0))
    _set_if_supported("nav_score_raw_sum", values.get("nav_score_raw_sum", 0.0))
    _set_if_supported("hover_score_raw_sum", values.get("hover_score_raw_sum", 0.0))
    _set_if_supported("wp_score_raw_sum", values.get("wp_score_raw_sum", 0.0))
    _set_if_supported("time_norm_sum", values.get("time_norm_sum", 0.0))
    _set_if_supported("smooth_norm_sum", values.get("smooth_norm_sum", 0.0))
    _set_if_supported("visited_waypoints_sum", values.get("visited_waypoints_sum", 0.0))
    _set_if_supported("total_waypoints_sum", values.get("total_waypoints_sum", 0.0))
    _set_if_supported("remaining_waypoints_sum", values.get("remaining_waypoints_sum", 0.0))
    _set_if_supported("arrival_success_sum", values.get("arrival_success_sum", 0.0))
    _set_if_supported("hover_success_sum", values.get("hover_success_sum", 0.0))
    _set_if_supported("hover_failed_sum", values.get("hover_failed_sum", 0.0))
    _set_if_supported("drone_radius_sum", values.get("drone_radius_sum", 0.0))
    _set_if_supported("arrival_radius_sum", values.get("arrival_radius_sum", 0.0))
    _set_if_supported("goal_radius_sum", values.get("goal_radius_sum", 0.0))
    _set_if_supported("waypoint_radius_sum", values.get("waypoint_radius_sum", 0.0))
    _set_if_supported("obstacle_radius_sum", values.get("obstacle_radius_sum", 0.0))
    _set_if_supported("obstacle_num_sum", values.get("obstacle_num_sum", 0.0))


def _strip_sum_fields(values):
    sanitized = dict(values)
    for field_name in EXPORT_SCORE_DETAIL_SUM_FIELDS:
        sanitized.pop(field_name, None)
    return sanitized


def build_end_info_message(game_id, end_info_dicts):
    valid_end_info_dicts = [item for item in end_info_dicts if isinstance(item, dict)]
    primary_end_info_dict = valid_end_info_dicts[0] if valid_end_info_dicts else {}
    export_score_detail_values = [
        _build_score_detail_values(env_idx, end_info_dict)
        for env_idx, end_info_dict in enumerate(valid_end_info_dicts)
    ]
    score_detail_values = _group_score_detail_values(export_score_detail_values)

    end_info_message = EndInfo(
        protocol_version=primary_end_info_dict.get("protocol_version", "v1"),
        project_code=primary_end_info_dict.get("project_code", "drone_obstacle_nav"),
        game_id=str(primary_end_info_dict.get("game_id", game_id)),
    )
    end_info_message.layout.CopyFrom(build_layout_message(primary_end_info_dict.get("layout", {})))

    aggregate_values = _build_aggregate_values(export_score_detail_values)
    _apply_score_values(end_info_message, aggregate_values)

    for detail_values in score_detail_values:
        detail_message = end_info_message.score_detail.add()
        _apply_score_values(detail_message, detail_values)

    for detail_values in export_score_detail_values:
        detail_message = end_info_message.export_score_detail.add()
        detail_message.env_index = detail_values["env_index"]
        _apply_score_values(detail_message, _strip_sum_fields(detail_values))

    return end_info_message


def proto_to_dict(message):
    json_kwargs = {"preserving_proto_field_name": True}
    try:
        json_kwargs["always_print_fields_with_no_presence"] = True
        raw_json = MessageToJson(message, **json_kwargs)
    except TypeError:
        json_kwargs.pop("always_print_fields_with_no_presence", None)
        json_kwargs["including_default_value_fields"] = True
        raw_json = MessageToJson(message, **json_kwargs)
    payload = json.loads(raw_json)

    # `score_detail` entries are aggregated across multiple envs, so `env_index`
    # is not meaningful there. When protobuf default fields are force-printed,
    # the unset integer shows up as `0`, which is misleading in exported JSON.
    if isinstance(message, EndInfo) and isinstance(payload, dict):
        score_detail = payload.get("score_detail")
        if isinstance(score_detail, list):
            for detail in score_detail:
                if isinstance(detail, dict):
                    detail.pop("env_index", None)

    return payload


def proto_to_compact_json(message):
    return json.dumps(proto_to_dict(message), ensure_ascii=False, separators=(",", ":"))
