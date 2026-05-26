#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import datetime
import json
import os
import subprocess
import re

from arena_proto.arena2plat_pb2 import CampInfo, GameData, GameStatus
from google.protobuf.timestamp_pb2 import Timestamp

from .proto_builders import (
    build_end_info_message,
    build_frames_message,
    build_start_info_message,
    build_trajectory_file_message,
    proto_to_compact_json,
    proto_to_dict,
)


def _write_json_file(file_path, payload):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as outfile:
        json.dump(payload, outfile, indent=4, ensure_ascii=False)


def _ensure_start_timestamp(drone_env):
    start_timestamp = getattr(drone_env, "start_timestamp", None)
    if start_timestamp is not None:
        return start_timestamp

    now = datetime.datetime.utcnow()
    start_timestamp = Timestamp()
    start_timestamp.FromDatetime(now)
    drone_env.start_timestamp = start_timestamp
    return start_timestamp


def _build_replay_task_type(end_info_dict, env_idx):
    if not isinstance(end_info_dict, dict):
        return f"env_{env_idx}"

    result_dict = end_info_dict.get("result", {}) if isinstance(end_info_dict.get("result"), dict) else {}
    layout_dict = end_info_dict.get("layout", {}) if isinstance(end_info_dict.get("layout"), dict) else {}

    task_type = str(result_dict.get("task_type") or "").strip()
    if not task_type:
        obstacle_num = int(float(layout_dict.get("num_obstacles", 0) or 0))
        obstacle_radius = int(round(float(layout_dict.get("radius", 0.0) or 0.0) * 100.0))
        waypoint_num = int(float(layout_dict.get("num_waypoints", 0) or 0))
        task_type = f"obs_{obstacle_num}_r_{obstacle_radius}_wp_{waypoint_num}"

    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", task_type).strip("_")
    return sanitized or f"env_{env_idx}"


def _write_replay_json_files(battle_dir, end_info_dicts):
    replay_file_names = []
    result_dir = f"{battle_dir}/result"
    for env_idx, end_info_dict in enumerate(end_info_dicts):
        replay_task_type = _build_replay_task_type(end_info_dict, env_idx)
        replay_file_name = f"env_{env_idx}_{replay_task_type}.json"
        replay_file_path = f"{result_dir}/{replay_file_name}"
        layout = end_info_dict.get("layout", {}) if isinstance(end_info_dict, dict) else {}
        trajectory = end_info_dict.get("trajectory", []) if isinstance(end_info_dict, dict) else []
        replay_file_message = build_trajectory_file_message(env_idx, trajectory, layout_dict=layout)
        _write_json_file(replay_file_path, proto_to_dict(replay_file_message))
        replay_file_names.append(replay_file_name)
    return replay_file_names
def finalize_result_artifacts(drone_env, end_info_dicts):
    result_dir = f"{drone_env.battle_dir}/result"
    os.makedirs(result_dir, exist_ok=True)

    start_timestamp = _ensure_start_timestamp(drone_env)

    start_info_message = build_start_info_message(drone_env.game_id)
    replay_file_names = _write_replay_json_files(drone_env.battle_dir, end_info_dicts)
    frames_message = build_frames_message()
    end_info_message = build_end_info_message(drone_env.game_id, end_info_dicts)

    # Reaching artifact finalization means the evaluation episode ended normally,
    # regardless of whether each env result is success / timeout / failed.
    drone_env.game_status = GameStatus.success

    camp = CampInfo(
        camp_type="blue",
        camp_code="A",
        start_info=proto_to_compact_json(start_info_message),
        end_info=proto_to_compact_json(end_info_message),
    )
    json_messages = proto_to_compact_json(frames_message)

    now = datetime.datetime.utcnow()
    end_timestamp = Timestamp()
    end_timestamp.FromDatetime(now)

    output = GameData(
        name=drone_env.game_id,
        project_code="drone_obstacle_nav",
        status=drone_env.game_status,
        camps=[camp],
        frames=json_messages,
        start_time=start_timestamp,
        end_time=end_timestamp,
    )

    mp4_dir = f"{drone_env.battle_dir}/mp4"
    if drone_env.enable_video and os.path.isdir(mp4_dir):
        try:
            drone_env.env.finalize_recording()
        except Exception as e:
            drone_env.logger.warning(f"[视频录制] 录制收尾失败: {e}")

        mp4_files = sorted(name for name in os.listdir(mp4_dir) if name.endswith(".mp4"))
        if mp4_files:
            zip_file = os.path.join(mp4_dir, f"{drone_env.game_id}.mp4.zip")
            subprocess.run(
                ["zip", "-j", zip_file, *mp4_files],
                cwd=mp4_dir,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    json_file = f"{result_dir}/{drone_env.game_id}.json"
    output_dict = proto_to_dict(output)
    output_dict["status"] = int(drone_env.game_status)
    _write_json_file(json_file, output_dict)

    done_file = f"{drone_env.battle_dir}/{drone_env.game_id}.done"
    with open(done_file, "w") as done:
        done.writelines("done")

    if drone_env.logger:
        drone_env.logger.info(
            f"json_file {json_file} create success, replay_files {replay_file_names}, done_file {done_file} create success"
        )
    else:
        print(
            f"json_file {json_file} create success, replay_files {replay_file_names}, done_file {done_file} create success"
        )


def finalize_error_result_artifacts(drone_env, status_message, game_status=GameStatus.error):
    result_dir = f"{drone_env.battle_dir}/result"
    os.makedirs(result_dir, exist_ok=True)

    start_timestamp = _ensure_start_timestamp(drone_env)
    frames_message = build_frames_message()
    start_info_message = build_start_info_message(drone_env.game_id)

    camp = CampInfo(
        camp_type="blue",
        camp_code="A",
        start_info=proto_to_compact_json(start_info_message),
        end_info="{}",
    )

    end_timestamp = Timestamp()
    end_timestamp.FromDatetime(datetime.datetime.utcnow())

    output = GameData(
        name=drone_env.game_id,
        project_code="drone_obstacle_nav",
        status=game_status,
        status_message=str(status_message),
        camps=[camp],
        frames=proto_to_compact_json(frames_message),
        start_time=start_timestamp,
        end_time=end_timestamp,
    )

    json_file = f"{result_dir}/{drone_env.game_id}.json"
    output_dict = proto_to_dict(output)
    output_dict["status"] = int(game_status)
    _write_json_file(json_file, output_dict)

    done_file = f"{drone_env.battle_dir}/{drone_env.game_id}.done"
    with open(done_file, "w") as done:
        done.writelines("done")

    drone_env.game_status = game_status

    if drone_env.logger:
        drone_env.logger.info(
            f"error json_file {json_file} create success, done_file {done_file} create success, status_message={status_message}"
        )
    else:
        print(
            f"error json_file {json_file} create success, done_file {done_file} create success, status_message={status_message}"
        )
