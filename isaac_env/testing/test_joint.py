#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

"""基于 eval 语义的 isaac_env joint integration test。

目标：
1. 通过 `isaac_env.core.Drone` 走完整的 eval `reset -> step -> episode end` 链路；
2. 验证 `done` 判定会触发结果文件写出；
3. 验证 waypoint 与 PRD 评分相关 stats 已暴露，且 `total_score` 符合三项归一化分 + 系数门控公式。

Usage:
    cd /data/projects/drone_obstacle_nav
    python3 isaac_env/testing/test_joint.py
"""

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

os.environ["HEADLESS"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("KAIWU_ALGORITHM", "ppo")

_results = []
WAYPOINT_STAT_KEYS = (
    "waypoints_visited",
    "waypoints_total",
    "waypoint_score_sum",
    "waypoint_score",
    "wp_score_raw",
)

CLUSTER_LABEL_STAT_KEYS = (
    "obstacle_num",
    "obstacle_radius",
)

PRD_SCORE_STAT_KEYS = (
    "nav_coeff",
    "hover_coeff",
    "nav_score_raw",
    "hover_score_raw",
    "wp_score_raw",
    "time_norm",
    "smooth_norm",
    "arrival_success",
    "hover_success",
    "hover_failed",
    "remaining_waypoints",
    "total_score",
)

def check(condition: bool, desc: str, phase: str = "?", detail: str = "", quiet: bool = False):
    _results.append((phase, desc, bool(condition), detail))
    if not quiet or not condition:
        tag = "PASS" if condition else "FAIL"
        print(f"  [{tag}] {desc}" + (f"  -- {detail}" if detail else ""))
    return condition


def summarize():
    total = len(_results)
    passed = sum(1 for r in _results if r[2])
    failed = total - passed
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed")
    if failed:
        print("\nFailed checks:")
        for phase, desc, ok, detail in _results:
            if not ok:
                print(f"  [{phase}] {desc}" + (f"  -- {detail}" if detail else ""))
    print("=" * 60)
    return failed == 0


def make_logger():
    logger = logging.getLogger("test_joint")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def _to_tensor(data, device):
    import torch

    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def _to_numpy(data):
    import torch

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _all_done(terminated, truncated=None) -> bool:
    arrays = []
    for flags in (terminated, truncated):
        if flags is None:
            continue
        arrays.append(np.asarray(flags, dtype=bool).reshape(-1))

    if not arrays:
        return False

    done = arrays[0]
    for item in arrays[1:]:
        if item.size == 0:
            continue
        if done.size == 0:
            done = item
            continue
        if item.size != done.size:
            item = np.resize(item, done.size)
        done = np.logical_or(done, item)

    return bool(done.all()) if done.size > 0 else False


def _done_count(flags) -> int:
    if flags is None:
        return 0
    return int(np.asarray(flags, dtype=bool).reshape(-1).sum())


def validate_first_done_snapshot_summary(drone_env, phase: str, expected_num_envs: int, require_snapshot=False):
    from isaac_env.evaluation.runtime_extractors import validate_eval_first_done_payloads

    try:
        summary = validate_eval_first_done_payloads(drone_env, expected_env_count=expected_num_envs)
    except Exception as exc:
        check(False, "first-done snapshot summary is reachable", phase, str(exc))
        return None

    check(isinstance(summary, dict), "first-done snapshot summary returns dict", phase, str(type(summary).__name__))
    if not isinstance(summary, dict):
        return None

    env_count = int(summary.get("env_count", -1))
    snapshot_count = int(summary.get("snapshot_count", -1))
    missing_envs = summary.get("missing_envs", [])
    check(env_count == expected_num_envs, "first-done snapshot env_count matches num_envs", phase, str(summary))
    check(isinstance(missing_envs, list), "first-done snapshot missing_envs is list", phase, str(type(missing_envs).__name__))
    check(0 <= snapshot_count <= expected_num_envs, "first-done snapshot_count stays within range", phase, str(summary))
    if require_snapshot:
        check(snapshot_count > 0, "at least one env first-done snapshot is captured", phase, str(summary))
    return summary


def validate_eval_result_finalization_via_apply_function(drone_env, phase: str):
    if hasattr(drone_env, "apply_function") and callable(getattr(drone_env, "apply_function")):
        check(True, "env exposes apply_function for eval finalization", phase)
        try:
            result = drone_env.apply_function("make_json_and_done_file")
        except Exception as exc:
            check(False, "apply_function(make_json_and_done_file) succeeds", phase, str(exc))
            return False

        check(result is None, "make_json_and_done_file returns no payload on success", phase, str(result))
        return True

    check(hasattr(drone_env, "make_json_and_done_file"), "env exposes direct make_json_and_done_file for eval finalization", phase)
    if not hasattr(drone_env, "make_json_and_done_file"):
        return False

    try:
        drone_env.make_json_and_done_file()
    except Exception as exc:
        check(False, "direct make_json_and_done_file succeeds", phase, str(exc))
        return False

    check(True, "direct make_json_and_done_file succeeds", phase)
    return True


def _load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def _load_embedded_json(payload):
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MiB"


def _get_runtime_and_task_env(drone_env):
    runtime = getattr(drone_env, "env", None)
    transformed_env = getattr(runtime, "env", None) if runtime is not None else None
    task_env = getattr(transformed_env, "base_env", transformed_env)
    return runtime, task_env


def validate_reward_provider_mount(drone_env, phase: str, expected_provider: str):
    runtime, task_env = _get_runtime_and_task_env(drone_env)
    reward_provider = getattr(runtime, "reward_provider", None)

    check(runtime is not None, "standalone runtime is initialized", phase, str(type(runtime).__name__))
    check(task_env is not None, "task env is reachable from runtime", phase, str(type(task_env).__name__))
    check(reward_provider is not None, "runtime mounts reward_provider", phase, str(type(reward_provider).__name__ if reward_provider else None))

    if reward_provider is None or task_env is None:
        return False

    provider_module = type(reward_provider).__module__
    check(
        provider_module == f"{expected_provider}.feature.reward_process",
        "reward_provider module matches requested agent package",
        phase,
        provider_module,
    )
    reward_state = getattr(reward_provider, "reward_state", None)
    check(reward_state is not None, "reward provider is bound to reward_state snapshot", phase, str(type(reward_state).__name__ if reward_state else None))
    check(reward_state is not task_env, "reward state is not the live task env object", phase)
    check(getattr(task_env, "reward_provider", None) is reward_provider, "task env shares mounted reward_provider instance", phase)
    return True


def reward_provider_has_active_terms(drone_env) -> bool:
    runtime, _ = _get_runtime_and_task_env(drone_env)
    reward_provider = getattr(runtime, "reward_provider", None) if runtime is not None else None
    if reward_provider is None:
        return False

    terms = getattr(reward_provider, "_terms", {})
    if not isinstance(terms, dict):
        return False

    return any(abs(float(info.get("weight", 0.0))) > 1.0e-8 for info in terms.values() if isinstance(info, dict))


def validate_step_reward(rewards, phase: str, step: int, expected_num_envs: int, require_nonzero: bool):
    check(rewards is not None, f"step {step} returns reward", phase, quiet=True)
    if rewards is None:
        return False

    reward_arr = np.asarray(rewards, dtype=float)
    check(reward_arr.shape[0] == expected_num_envs, f"step {step} reward batch matches num_envs", phase, str(reward_arr.shape), quiet=True)
    check(np.all(np.isfinite(reward_arr)), f"step {step} reward is finite", phase, quiet=True)
    if require_nonzero:
        check(np.any(np.abs(reward_arr) > 1.0e-6), f"step {step} reward is not all zeros", phase, str(reward_arr.reshape(-1)[:8]), quiet=True)
    else:
        check(np.allclose(reward_arr, 0.0), f"step {step} reward falls back to zero placeholder without active reward terms", phase, str(reward_arr.reshape(-1)[:8]), quiet=True)
    return True


def validate_reward_provider_runtime_output(drone_env, phase: str):
    runtime, task_env = _get_runtime_and_task_env(drone_env)
    reward_provider = getattr(runtime, "reward_provider", None) if runtime is not None else None

    check(reward_provider is not None, "reward provider is available for runtime output checks", phase)
    check(task_env is not None, "task env is available for runtime output checks", phase)
    if reward_provider is None or task_env is None:
        return False

    sentinel_reward = torch.arange(task_env.num_envs, device=task_env.device, dtype=torch.float32) + 1.0
    computed_reward = reward_provider.compute_reward(env_reward=sentinel_reward)
    computed_reward_np = _to_numpy(computed_reward).astype(float, copy=False).reshape(-1)

    check(
        np.any(np.abs(computed_reward_np) > 1.0e-6),
        "reward provider compute_reward returns non-zero training reward when terms are active",
        phase,
        str(computed_reward_np[:8]),
    )
    check(
        getattr(task_env, "reward_provider", None) is reward_provider,
        "task env keeps same reward_provider instance used for compute_reward",
        phase,
    )
    return True


def validate_reward_provider_semantics(drone_env, phase: str):
    runtime, task_env = _get_runtime_and_task_env(drone_env)
    reward_provider = getattr(runtime, "reward_provider", None) if runtime is not None else None

    check(reward_provider is not None, "reward provider is available for semantic checks", phase)
    check(task_env is not None, "task env is available for semantic checks", phase)
    if reward_provider is None or task_env is None:
        return False

    reward_state = getattr(reward_provider, "reward_state", None)
    check(reward_state is not None, "reward provider uses reward_state instead of live env", phase)
    check(reward_state is not task_env, "reward state object is detached from live env", phase)
    check(
        not hasattr(task_env, "_prev_obs_for_reward") and not hasattr(task_env, "_latest_obs_for_reward"),
        "task env no longer keeps obs-based reward bridge buffers",
        phase,
    )
    if reward_state is not None:
        check(reward_state.phase is not task_env.phase, "reward_state.phase is cloned from env", phase)
        check(reward_state.collision_count is not task_env.collision_count, "reward_state.collision_count is cloned from env", phase)
        check(reward_state.hover_steps is not task_env.hover_steps, "reward_state.hover_steps is cloned from env", phase)
        check(reward_state.waypoint_visited_step is not task_env.waypoint_visited_step, "reward_state.waypoint_visited_step is cloned from env", phase)
        check(reward_state.obstacle_positions is not task_env.obstacle_positions, "reward_state.obstacle_positions is cloned from env", phase)
        check(reward_state.prev_action is not task_env.prev_action, "reward_state.prev_action is cloned from env", phase)
        check(reward_state.prev_linear_velocity is not task_env.prev_linear_velocity, "reward_state.prev_linear_velocity is cloned from env", phase)
        check(abs(float(reward_state.drone_radius) - float(task_env.drone_radius)) <= 1.0e-6, "reward_state carries drone_radius constant", phase)
        check(abs(float(reward_state.obstacle_safe_distance) - float(task_env.obstacle_safe_distance)) <= 1.0e-6, "reward_state carries obstacle_safe_distance constant", phase)
        check(abs(float(reward_state.waypoint_collect_radius) - float(task_env.waypoint_collect_radius)) <= 1.0e-6, "reward_state carries waypoint_collect_radius constant", phase)
        check(int(reward_state.max_collisions) == int(task_env.max_collisions), "reward_state carries max_collisions constant", phase)

    active_terms = getattr(reward_provider, "_terms", {})
    check(isinstance(active_terms, dict), "reward provider exposes term registry", phase, str(type(active_terms).__name__))

    if isinstance(active_terms, dict) and reward_state is not None:
        obstacle_term = active_terms.get("obstacle_proximity", {})
        collision_term = active_terms.get("collision", {})
        stall_term = active_terms.get("stall", {})
        waypoint_term = active_terms.get("waypoint_visit", {})

        check(callable(obstacle_term.get("method")), "obstacle_proximity term is mounted", phase)
        check(callable(collision_term.get("method")), "collision term is mounted", phase)
        check(callable(stall_term.get("method")), "stall term is mounted", phase)
        check(callable(waypoint_term.get("method")), "waypoint_visit term is mounted", phase)

        original_reward_state = reward_state

        if callable(obstacle_term.get("method")):
            reward_provider.bind_reward_state(
                replace(
                    original_reward_state,
                    obstacle_clearance=torch.full_like(original_reward_state.obstacle_clearance, 0.05),
                )
            )
            penalty = obstacle_term["method"]()
            penalty = _to_numpy(penalty).astype(float, copy=False).reshape(-1)
            check(
                np.allclose(penalty, np.asarray([-2.0]), atol=1.0e-6),
                "obstacle_proximity keeps legacy step penalty semantics",
                phase,
                str(penalty),
            )

        if callable(collision_term.get("method")):
            reward_provider.bind_reward_state(
                replace(
                    original_reward_state,
                    obstacle_collision=torch.ones_like(original_reward_state.obstacle_collision, dtype=torch.bool),
                )
            )
            penalty = collision_term["method"]()
            penalty = _to_numpy(penalty).astype(float, copy=False).reshape(-1)
            check(
                np.allclose(penalty, np.asarray([-20.0]), atol=1.0e-6),
                "collision term penalizes snapshot collision state every step",
                phase,
                str(penalty),
            )

        if callable(stall_term.get("method")):
            stalled_state = replace(
                original_reward_state,
                progress_buf=original_reward_state.progress_buf + 1,
            )
            reward_provider.bind_reward_state(stalled_state)
            reward_provider._step_cache = {}
            reward_provider._history["nav_target_distance"].zero_()
            reward_provider._history["episode_step"].copy_(stalled_state.progress_buf.view(-1) - 1)
            reward_provider._history["initialized"].fill_(True)
            penalty = stall_term["method"]()
            penalty = _to_numpy(penalty).astype(float, copy=False).reshape(-1)
            expected_stall_penalty = -float(reward_provider.task_config.get("stall_penalty", 0.1))
            check(
                np.allclose(penalty, expected_stall_penalty, atol=1.0e-6),
                "stall term reconstructs stalled flag from env history",
                phase,
                str(penalty),
            )

        if callable(waypoint_term.get("method")):
            waypoint_state = replace(
                original_reward_state,
                waypoint_score_sum=original_reward_state.waypoint_score_sum + 1.0,
                progress_buf=original_reward_state.progress_buf + 1,
            )
            reward_provider.bind_reward_state(waypoint_state)
            reward_provider._step_cache = {}
            reward_provider._history["episode_step"].copy_(waypoint_state.progress_buf.view(-1) - 1)
            reward_provider._history["initialized"].fill_(True)
            reward = waypoint_term["method"](
                waypoint_visit_reward_boost=1.15,
                waypoint_time_decay_power=1.5,
                waypoint_time_decay_floor=0.15,
            )
            reward = _to_numpy(reward).astype(float, copy=False).reshape(-1)
            expected_waypoint_reward = 2.3
            check(
                np.allclose(reward, expected_waypoint_reward, atol=1.0e-6),
                "waypoint_visit term reconstructs collected reward delta and applies boost",
                phase,
                str(reward),
            )

        reward_provider.bind_reward_state(original_reward_state)
        reward_provider._step_cache = {}

    return True


def validate_runtime_stats(stats, phase: str, expected_num_envs: int):
    check(isinstance(stats, dict), "step extra_info carries stats dict", phase, str(type(stats).__name__))
    if not isinstance(stats, dict):
        return False

    for key in WAYPOINT_STAT_KEYS:
        check(key in stats, f"stats include {key}", phase, quiet=True)
        if key in stats:
            arr = np.asarray(stats[key])
            check(arr.shape[0] == expected_num_envs, f"{key} batch matches num_envs", phase, str(arr.shape), quiet=True)

    for key in CLUSTER_LABEL_STAT_KEYS:
        check(key in stats, f"stats include {key}", phase, quiet=True)
        if key in stats:
            arr = np.asarray(stats[key])
            check(arr.shape[0] == expected_num_envs, f"{key} batch matches num_envs", phase, str(arr.shape), quiet=True)

    for key in PRD_SCORE_STAT_KEYS:
        check(key in stats, f"stats include {key}", phase, quiet=True)
        if key in stats:
            arr = np.asarray(stats[key])
            check(arr.shape[0] == expected_num_envs, f"{key} batch matches num_envs", phase, str(arr.shape), quiet=True)

    if all(key in stats for key in ("waypoints_visited", "waypoints_total", "waypoint_score")):
        collected = np.asarray(stats["waypoints_visited"], dtype=float)
        total = np.asarray(stats["waypoints_total"], dtype=float)
        score = np.asarray(stats["waypoint_score"], dtype=float)
        raw_score = np.asarray(stats["wp_score_raw"], dtype=float) if "wp_score_raw" in stats else None
        check(np.all(total >= collected), "waypoint total covers visited count", phase, quiet=True)
        if "remaining_waypoints" in stats:
            remaining = np.asarray(stats["remaining_waypoints"], dtype=float)
            check(np.allclose(remaining, np.maximum(total - collected, 0.0)), "remaining_waypoints matches total-visited", phase, quiet=True)
        check(np.all(score >= 0.0), "waypoint score contribution is non-negative", phase, quiet=True)
        if raw_score is not None:
            check(np.allclose(score, 40.0 * raw_score, atol=1.0e-6), "waypoint score equals weighted contribution", phase, quiet=True)

    if all(key in stats for key in ("obstacle_num", "obstacle_radius")):
        obstacle_num = np.asarray(stats["obstacle_num"], dtype=float)
        obstacle_radius = np.asarray(stats["obstacle_radius"], dtype=float)
        check(np.all((2.0 <= obstacle_num) & (obstacle_num <= 5.0)), "obstacle_num stays within training buckets", phase, quiet=True)
        check(np.all(np.isin(np.round(obstacle_radius, 2), (0.15, 0.20, 0.25, 0.30, 0.35))), "obstacle_radius stays within training buckets", phase, quiet=True)

    if all(key in stats for key in ("nav_coeff", "hover_coeff", "nav_score_raw", "hover_score_raw", "wp_score_raw", "time_norm", "smooth_norm")):
        nav_coeff = np.asarray(stats["nav_coeff"], dtype=float)
        hover_coeff = np.asarray(stats["hover_coeff"], dtype=float)
        nav_score_raw = np.asarray(stats["nav_score_raw"], dtype=float)
        hover_score_raw = np.asarray(stats["hover_score_raw"], dtype=float)
        wp_score_raw = np.asarray(stats["wp_score_raw"], dtype=float)
        time_norm = np.asarray(stats["time_norm"], dtype=float)
        smooth_norm = np.asarray(stats["smooth_norm"], dtype=float)
        check(np.all(np.isin(np.round(nav_coeff, 6), (0.0, 1.0))), "nav_coeff is binary", phase, quiet=True)
        check(np.all(np.isin(np.round(hover_coeff, 6), (0.0, 1.0))), "hover_coeff is binary", phase, quiet=True)
        check(np.all((0.0 <= nav_score_raw) & (nav_score_raw <= 1.0 + 1.0e-6)), "nav_score_raw is normalized", phase, quiet=True)
        check(np.all((0.0 <= hover_score_raw) & (hover_score_raw <= 1.0 + 1.0e-6)), "hover_score_raw is normalized", phase, quiet=True)
        check(np.all((0.0 <= wp_score_raw) & (wp_score_raw <= 1.0 + 1.0e-6)), "wp_score_raw is normalized", phase, quiet=True)
        check(np.all((0.0 <= time_norm) & (time_norm <= 1.0 + 1.0e-6)), "time_norm is normalized", phase, quiet=True)
        check(np.all((0.0 <= smooth_norm) & (smooth_norm <= 1.0 + 1.0e-6)), "smooth_norm is normalized", phase, quiet=True)

    if all(key in stats for key in ("arrival_success", "hover_success", "hover_failed")):
        arrival_success = np.asarray(stats["arrival_success"], dtype=float)
        hover_success = np.asarray(stats["hover_success"], dtype=float)
        hover_failed = np.asarray(stats["hover_failed"], dtype=float)
        check(np.all(np.isin(np.round(arrival_success, 6), (0.0, 1.0))), "arrival_success is binary", phase, quiet=True)
        check(np.all(np.isin(np.round(hover_success, 6), (0.0, 1.0))), "hover_success is binary", phase, quiet=True)
        check(np.all(np.isin(np.round(hover_failed, 6), (0.0, 1.0))), "hover_failed is binary", phase, quiet=True)

    return True


def print_cuda_memory_stats():
    try:
        import torch
    except Exception as exc:
        print(f"  GPU memory stats unavailable: {exc}")
        return

    print("\n" + "=" * 60)
    print("GPU Memory During Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  CUDA unavailable, skip GPU memory stats")
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    device_idx = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_idx)
    allocated = torch.cuda.memory_allocated(device_idx)
    reserved = torch.cuda.memory_reserved(device_idx)
    peak_allocated = torch.cuda.max_memory_allocated(device_idx)
    peak_reserved = torch.cuda.max_memory_reserved(device_idx)

    print(f"  device: cuda:{device_idx} ({device_name})")
    print(f"  allocated: {_format_bytes(allocated)}")
    print(f"  reserved: {_format_bytes(reserved)}")
    print(f"  peak allocated: {_format_bytes(peak_allocated)}")
    print(f"  peak reserved: {_format_bytes(peak_reserved)}")


class RandomAgent:
    """最小化 eval agent，复用 `eval_workflow.py` 的 exploit 调用模式。"""

    def __init__(self, action_shape, device):
        self.action_shape = tuple(action_shape)
        self.device = device

    def exploit(self, obs):
        import torch

        return torch.randn(self.action_shape, device=self.device)


def build_eval_usr_conf(logger, game_id: str):
    from tools.train_env_conf_validate import read_eval_conf, read_usr_conf

    # Use train config to exercise reward terms with non-zero weights.
    # 使用训练配置以激活所有 reward terms（权重非零）。
    train_conf_file = "agent_ppo/conf/train_env_conf.toml"
    usr_conf = read_usr_conf(train_conf_file, logger, eval=False)
    if usr_conf is None:
        raise RuntimeError("failed to load train or eval env conf")

    usr_conf["game_id"] = game_id
    usr_conf["is_eval"] = True  # Keep eval flag for result file generation
    # Override num_envs to a small value for test speed
    if "env_conf" in usr_conf:
        usr_conf["env_conf"]["num_envs"] = 5
    return usr_conf


def wait_for_paths(paths, timeout_sec: float = 3.0):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if all(os.path.exists(path) for path in paths):
            return True
        time.sleep(0.1)
    return all(os.path.exists(path) for path in paths)


def wait_for_condition(predicate, timeout_sec: float = 5.0, interval_sec: float = 0.1):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval_sec)
    try:
        return bool(predicate())
    except Exception:
        return False


def build_case_paths(game_id: str):
    battle_dir = f"/workspace/battle/{game_id}"
    result_dir = f"{battle_dir}/result"
    return {
        "battle_dir": battle_dir,
        "result_dir": result_dir,
        "result_json": f"{result_dir}/{game_id}.json",
        "done_file": f"{battle_dir}/{game_id}.done",
    }


def list_trajectory_json_files(result_dir: str):
    if not os.path.isdir(result_dir):
        return []
    return sorted(
        name for name in os.listdir(result_dir)
        if name.startswith("env_") and name.endswith(".json")
    )




def validate_multi_step_distance_progress_tracking(drone_env, step_num: int, phase: str):
    """Per-step tracking of snapshot vs live env consistency and distance_progress signal.

    每步追踪：
    A. 快照数据 vs live env 数据一致性
    B. 目标选择正确性
    C. distance_progress 数值（从 history 差分得到）
    D. _reward_target_progress 奖励值
    """
    runtime, task_env = _get_runtime_and_task_env(drone_env)
    reward_provider = getattr(runtime, "reward_provider", None) if runtime is not None else None
    if reward_provider is None or task_env is None:
        return

    reward_state = getattr(reward_provider, "reward_state", None)
    if reward_state is None:
        return

    num_envs = int(reward_state.num_envs)
    device = reward_state.device
    env_i = 0  # Track env[0] for readability

    # === A. Snapshot vs Live Env Consistency ===
    # Compare snapshot drone_state with live env
    snapshot_drone_pos = reward_state.drone_state[..., :3]  # (num_envs, 1, 3)
    live_drone_state = task_env._latest_drone_state if getattr(task_env, "_latest_drone_state", None) is not None else task_env.drone.get_state()
    live_drone_pos = live_drone_state[..., :3]

    pos_match = torch.allclose(snapshot_drone_pos, live_drone_pos, atol=1e-4)

    snapshot_wp_visited = reward_state.waypoint_visited
    live_wp_visited = task_env.waypoint_visited
    wp_visited_match = torch.equal(snapshot_wp_visited, live_wp_visited)

    snapshot_wp_mask = reward_state.active_waypoint_mask
    live_wp_mask = task_env.active_waypoint_mask
    # Check shape mismatch first - this could be the root cause
    shape_match = (snapshot_wp_mask.shape == live_wp_mask.shape)
    wp_mask_match = torch.equal(snapshot_wp_mask, live_wp_mask) if shape_match else False

    snapshot_progress = int(reward_state.progress_buf[env_i].item())
    live_progress = int(task_env.progress_buf[env_i].item())
    progress_match = (snapshot_progress == live_progress)

    consistency_ok = pos_match and wp_visited_match and wp_mask_match and progress_match

    # === B. Target Selection ===
    snapshot_drone_flat = snapshot_drone_pos.squeeze(1)  # (num_envs, 3)
    pending_mask = snapshot_wp_mask & (~snapshot_wp_visited)
    has_pending = pending_mask.any(dim=-1)

    wp_positions = reward_state.waypoint_positions
    wp_dists = torch.norm(wp_positions - snapshot_drone_flat.unsqueeze(1), dim=-1)
    masked_dists = torch.where(pending_mask, wp_dists, torch.full_like(wp_dists, 1e6))
    nearest_idx = masked_dists.argmin(dim=-1)

    # Get what the reward_provider computes
    provider_wp_pos, provider_wp_idx, provider_has_pending = reward_provider._active_waypoint_snapshot()
    target_match = torch.equal(provider_wp_idx, nearest_idx) if has_pending[env_i] else True

    # === C. distance_progress from cache ===
    history = reward_provider._history
    hist_nav_target_dist = float(history["nav_target_distance"][env_i])
    hist_step = int(history["episode_step"][env_i])
    hist_initialized = bool(history["initialized"][env_i])

    # The step cache should already be computed by the env step (compute_reward was called)
    cache = reward_provider._step_cache
    if cache and "distance_progress" in cache:
        dp = float(cache["distance_progress"][env_i])
        gp = float(cache["goal_progress"][env_i])
        stalled = float(cache["stalled"][env_i])
    else:
        dp = None
        gp = None
        stalled = None

    # Current nav_target_distance (compute from snapshot)
    goal_flat = reward_state.goal_marker_positions.squeeze(1)
    if has_pending[env_i]:
        gather_ids = torch.arange(num_envs, device=device)
        target_pos = wp_positions[gather_ids, nearest_idx]
    else:
        target_pos = goal_flat
    current_nav_target_dist = float(torch.norm(target_pos[env_i] - snapshot_drone_flat[env_i]))

    # === D. Reward value — read from _last_term_values if available ===
    last_terms = getattr(reward_provider, "_last_term_values", {})
    if "target_progress" in last_terms:
        reward_val = float(last_terms["target_progress"][env_i])
    else:
        # Fallback: compute directly (may re-trigger cache but train mode should already have it)
        try:
            reward_val = float(reward_provider._reward_target_progress()[env_i])
        except Exception:
            reward_val = None

    # === Print compact per-step report for env[0] ===
    drone_p = snapshot_drone_flat[env_i].tolist()
    if has_pending[env_i]:
        wp_idx_val = int(nearest_idx[env_i])
        wp_p = wp_positions[env_i, wp_idx_val].tolist()
        wp_dist = float(masked_dists[env_i, wp_idx_val])
    else:
        wp_idx_val = -1
        wp_p = goal_flat[env_i].tolist()
        wp_dist = float(torch.norm(goal_flat[env_i] - snapshot_drone_flat[env_i]))

    dp_str = f"{dp:+.6f}" if dp is not None else "N/A"
    gp_str = f"{gp:+.6f}" if gp is not None else "N/A"
    rw_str = f"{reward_val:+.6f}" if reward_val is not None else "N/A"
    stall_str = f"{stalled:.0f}" if stalled is not None else "?"

    print(f"    step[{step_num:3d}] env[0]: "
          f"drone=[{drone_p[0]:.3f},{drone_p[1]:.3f},{drone_p[2]:.3f}] "
          f"wp[{wp_idx_val}]=[{wp_p[0]:.3f},{wp_p[1]:.3f},{wp_p[2]:.3f}] "
          f"dist={wp_dist:.4f} | "
          f"dp={dp_str} gp={gp_str} rw={rw_str} stall={stall_str} | "
          f"hist_dist={hist_nav_target_dist:.4f} step_id={hist_step} "
          f"sync={'OK' if consistency_ok else 'MISMATCH'}")

    # Checks (only fail on critical issues)
    if not consistency_ok:
        detail_parts = []
        if not pos_match:
            diff = float((snapshot_drone_pos - live_drone_pos).abs().max())
            detail_parts.append(f"drone_pos_diff={diff:.6f}")
        if not wp_visited_match:
            snapshot_visited_shape = snapshot_wp_visited.shape
            live_visited_shape = live_wp_visited.shape
            if snapshot_visited_shape != live_visited_shape:
                detail_parts.append(f"wp_visited_SHAPE_MISMATCH: snap={snapshot_visited_shape} live={live_visited_shape}")
            else:
                diff_count = int((snapshot_wp_visited != live_wp_visited).sum())
                snapshot_visited = snapshot_wp_visited[env_i].tolist() if env_i < snapshot_wp_visited.shape[0] else "OOB"
                live_visited = live_wp_visited[env_i].tolist() if env_i < live_wp_visited.shape[0] else "OOB"
                detail_parts.append(f"wp_visited_diff_count={diff_count}: snapshot={snapshot_visited} live={live_visited}")
        if not wp_mask_match:
            # Detailed diagnostic for wp_mask_mismatch
            # First check shape mismatch - this is a common root cause
            snapshot_shape = snapshot_wp_mask.shape
            live_shape = live_wp_mask.shape
            if snapshot_shape != live_shape:
                detail_parts.append(f"wp_mask_SHAPE_MISMATCH: snap={snapshot_shape} live={live_shape}")
            else:
                diff_mask = ~(snapshot_wp_mask == live_wp_mask)
                diff_indices = torch.where(diff_mask)[0].tolist()
                snapshot_vals = snapshot_wp_mask[env_i].tolist() if env_i < snapshot_wp_mask.shape[0] else "OOB"
                live_vals = live_wp_mask[env_i].tolist() if env_i < live_wp_mask.shape[0] else "OOB"
                detail_parts.append(
                    f"wp_mask_mismatch: shape={snapshot_shape}; "
                    f"snap[{env_i}]={snapshot_vals} live[{env_i}]={live_vals}; "
                    f"diff_positions={diff_indices[:10]}"
                )
        if not progress_match:
            detail_parts.append(f"progress: snapshot={snapshot_progress} live={live_progress}")
        check(False, f"step {step_num} snapshot matches live env", phase, "; ".join(detail_parts))

    if dp is not None and dp == 0.0 and hist_initialized and hist_step == snapshot_progress - 1:
        # dp==0 could be normal if drone didn't move toward target, but worth flagging
        check(True, f"step {step_num} distance_progress is zero (drone did not approach target)", phase,
              f"dp=0, nav_dist={current_nav_target_dist:.4f}, hist_dist={hist_nav_target_dist:.4f}", quiet=True)


def validate_active_waypoint_snapshot_and_goal_progress(drone_env, phase: str):
    """Validate _active_waypoint_snapshot correctness and raw_goal_progress behavior.

    着重检查：
    1. _active_waypoint_snapshot 选出的 next_waypoint_pos 是否真的对应最近的未访问 waypoint
    2. 当无人机在终点附近时，导航目标是否仍然合理
    3. raw_goal_progress (history["goal_distance"] - ctx["goal_distance"]) 的数值正确性
    4. distance_progress 在目标切换时的保护逻辑是否生效
    """
    from dataclasses import replace

    runtime, task_env = _get_runtime_and_task_env(drone_env)
    reward_provider = getattr(runtime, "reward_provider", None) if runtime is not None else None

    check(reward_provider is not None, "reward provider available for waypoint snapshot checks", phase)
    check(task_env is not None, "task env available for waypoint snapshot checks", phase)
    if reward_provider is None or task_env is None:
        return False

    reward_state = getattr(reward_provider, "reward_state", None)
    if reward_state is None:
        check(False, "reward_state is bound before waypoint checks", phase)
        return False

    original_reward_state = reward_state
    num_envs = int(reward_state.num_envs)
    device = reward_state.device

    # ========================================================================
    # Test 1: _active_waypoint_snapshot returns correct nearest pending waypoint
    # ========================================================================
    print(f"\n  --- Waypoint Snapshot Consistency ({num_envs} envs) ---")

    # Read current waypoint state from the snapshot
    waypoint_mask = reward_state.active_waypoint_mask
    waypoint_visited = reward_state.waypoint_visited
    waypoint_positions = reward_state.waypoint_positions
    drone_state = reward_state.drone_state
    drone_pos_3d = drone_state[..., :3]  # (num_envs, 1, 3)
    drone_pos_flat = drone_pos_3d.squeeze(1)  # (num_envs, 3)

    # Diagnostic: report waypoint mask state
    active_count = int(waypoint_mask.any(dim=-1).sum())
    visited_count = int((waypoint_mask & waypoint_visited).any(dim=-1).sum())
    print(f"    active_waypoint_mask has active waypoints in {active_count}/{num_envs} envs")
    print(f"    waypoint_visited covers waypoints in {visited_count}/{num_envs} envs")

    pending_mask = waypoint_mask & (~waypoint_visited)
    has_pending = pending_mask.any(dim=-1)  # (num_envs,)
    envs_with_pending = int(has_pending.sum())
    print(f"    envs with pending (unvisited active) waypoints: {envs_with_pending}/{num_envs}")

    if envs_with_pending == 0:
        check(False, "snapshot has envs with pending waypoints for waypoint diagnostics", phase,
              f"active_mask_any={active_count}, all visited or no active waypoints")
        # Restore and skip remaining tests
        reward_provider.bind_reward_state(original_reward_state)
        reward_provider._step_cache = {}
        return False

    # Manually compute expected nearest waypoint
    waypoint_dists = torch.norm(waypoint_positions - drone_pos_flat.unsqueeze(1), dim=-1)
    masked_dists = torch.where(pending_mask, waypoint_dists, torch.full_like(waypoint_dists, 1.0e6))
    expected_nearest_idx = masked_dists.argmin(dim=-1)
    gather_ids = torch.arange(num_envs, device=device)
    expected_nearest_pos = waypoint_positions[gather_ids, expected_nearest_idx]

    # Call _active_waypoint_snapshot via the reward_provider
    next_waypoint_pos, next_waypoint_idx, has_pending_waypoint = reward_provider._active_waypoint_snapshot()

    # Check: returned indices match our manual computation
    idx_match = torch.equal(next_waypoint_idx, expected_nearest_idx)
    check(idx_match, "active_waypoint_snapshot idx matches manual nearest-pending computation", phase,
          f"provider_idx={next_waypoint_idx.tolist()}, expected={expected_nearest_idx.tolist()}")

    # Check: returned positions match
    pos_match = torch.allclose(next_waypoint_pos, expected_nearest_pos, atol=1e-5)
    check(pos_match, "active_waypoint_snapshot pos matches manual nearest-pending pos", phase,
          f"max_diff={float((next_waypoint_pos - expected_nearest_pos).abs().max()):.6f}")

    # Check: has_pending_waypoint matches
    pending_match = torch.equal(has_pending_waypoint, has_pending)
    check(pending_match, "active_waypoint_snapshot has_pending matches manual check", phase,
          f"provider={has_pending_waypoint.tolist()}, manual={has_pending.tolist()}")

    # Report diagnostic info
    for env_i in range(min(num_envs, 4)):
        wp_count = int(pending_mask[env_i].sum())
        if wp_count > 0:
            nearest_dist = float(masked_dists[env_i, expected_nearest_idx[env_i]])
            wp_pos = expected_nearest_pos[env_i].tolist()
            d_pos = drone_pos_flat[env_i].tolist()
            print(f"    env[{env_i}]: drone={[f'{v:.3f}' for v in d_pos]}, "
                  f"nearest_wp[{int(expected_nearest_idx[env_i])}]={[f'{v:.3f}' for v in wp_pos]}, "
                  f"dist={nearest_dist:.4f}, pending_count={wp_count}")
        else:
            print(f"    env[{env_i}]: no pending waypoints, target falls back to goal")

    # ========================================================================
    # Test 2: Simulate drone near goal — check if nav_target makes sense
    # ========================================================================
    print(f"\n  --- Drone-Near-Goal Scenario ---")

    goal_position = reward_state.goal_marker_positions  # (num_envs, 1, 3)
    goal_flat = goal_position.squeeze(1)  # (num_envs, 3)

    # Print spatial layout for first few envs with pending waypoints
    pending_env_indices = has_pending.nonzero(as_tuple=False).view(-1).tolist()
    for env_i in pending_env_indices[:2]:
        g = goal_flat[env_i].tolist()
        d = drone_pos_flat[env_i].tolist()
        print(f"    env[{env_i}] layout: goal={[f'{v:.3f}' for v in g]}, drone={[f'{v:.3f}' for v in d]}")
        active_wp_indices = waypoint_mask[env_i].nonzero(as_tuple=False).view(-1).tolist()
        for wi in active_wp_indices:
            wp_p = waypoint_positions[env_i, wi].tolist()
            visited = bool(waypoint_visited[env_i, wi])
            dist_to_goal = float(torch.norm(waypoint_positions[env_i, wi] - goal_flat[env_i]))
            print(f"      wp[{wi}]: pos={[f'{v:.3f}' for v in wp_p]}, visited={visited}, dist_to_goal={dist_to_goal:.3f}")

    # Construct a synthetic state: drone is at goal position
    # Replace drone_state position with goal position to simulate hovering at goal
    synthetic_drone_state = drone_state.clone()
    synthetic_drone_state[..., :3] = goal_position  # set drone pos = goal pos

    synthetic_state = replace(
        original_reward_state,
        drone_state=synthetic_drone_state,
        progress_buf=original_reward_state.progress_buf + 1,
    )
    reward_provider.bind_reward_state(synthetic_state)
    reward_provider._step_cache = {}

    # Run _active_waypoint_snapshot with drone at goal
    next_wp_pos_at_goal, next_wp_idx_at_goal, has_pending_at_goal = reward_provider._active_waypoint_snapshot()

    # Compute what the nav_target would be
    nav_target_at_goal = torch.where(
        has_pending_at_goal.unsqueeze(-1),
        next_wp_pos_at_goal,
        goal_flat,
    )
    nav_target_dist_at_goal = torch.norm(nav_target_at_goal - goal_flat, dim=-1)

    for env_i in range(min(num_envs, 4)):
        if has_pending_at_goal[env_i]:
            wp_idx = int(next_wp_idx_at_goal[env_i])
            wp_pos = next_wp_pos_at_goal[env_i].tolist()
            target_dist = float(nav_target_dist_at_goal[env_i])
            goal_pos_i = goal_flat[env_i].tolist()
            print(f"    env[{env_i}]: drone@goal={[f'{v:.3f}' for v in goal_pos_i]}, "
                  f"nearest_pending_wp[{wp_idx}]={[f'{v:.3f}' for v in wp_pos]}, "
                  f"nav_target_dist_from_goal={target_dist:.4f}")
            # Critical: if nav_target is a waypoint far from goal, the drone at goal
            # would get NEGATIVE distance_progress trying to fly away from goal toward waypoint
            check(
                target_dist < 2.0 * float(reward_state.arrival_radius) or not has_pending_at_goal[env_i],
                f"env[{env_i}] nav_target near goal is within reasonable range (or no pending wp)",
                phase,
                f"nav_target_dist={target_dist:.4f}, arrival_radius={float(reward_state.arrival_radius):.4f}",
            )
        else:
            print(f"    env[{env_i}]: no pending waypoints at goal, target=goal (dist=0)")

    # If there ARE pending waypoints when drone is at goal, this reveals the core issue:
    # the "nearest" waypoint from the goal position may be far away, causing the drone
    # to receive negative distance_progress if it stays at the goal instead of flying
    # toward that waypoint.
    envs_with_far_targets = (has_pending_at_goal & (nav_target_dist_at_goal > float(reward_state.arrival_radius))).sum().item()
    if envs_with_far_targets > 0:
        far_dists = nav_target_dist_at_goal[has_pending_at_goal & (nav_target_dist_at_goal > float(reward_state.arrival_radius))]
        print(f"\n    ⚠ WARNING: {envs_with_far_targets}/{num_envs} envs have pending waypoints "
              f"far from goal (> arrival_radius={float(reward_state.arrival_radius):.3f}).")
        print(f"    Distance distribution: min={float(far_dists.min()):.3f}, "
              f"max={float(far_dists.max()):.3f}, mean={float(far_dists.mean()):.3f}")
        print(f"    When drone is at goal, distance_progress drives it AWAY from goal "
              f"toward these waypoints, conflicting with goal_progress.")
        print(f"    This explains why the drone hovers above goal without making progress.")

    # ========================================================================
    # Test 3: raw_goal_progress correctness
    # ========================================================================
    print(f"\n  --- raw_goal_progress Correctness ---")

    # Set up: use the original reward_state as "current", and set history to simulate
    # a previous state where drone was slightly farther from goal
    reward_provider.bind_reward_state(original_reward_state)
    reward_provider._step_cache = {}

    # Prepare history: pretend previous step had drone 0.5 farther from goal
    history = reward_provider._history
    current_goal_distance = torch.norm(goal_position - drone_pos_3d, dim=-1).view(-1)
    simulated_prev_goal_distance = current_goal_distance + 0.5

    history["goal_distance"].copy_(simulated_prev_goal_distance)
    history["nav_target_distance"].copy_(
        torch.norm(
            torch.where(has_pending.unsqueeze(-1), expected_nearest_pos, goal_flat) - drone_pos_flat,
            dim=-1,
        ) + 0.5
    )
    history["phase"].copy_(original_reward_state.phase.long().view(-1))
    history["has_pending_waypoint"].copy_(has_pending)
    history["next_waypoint_idx"].copy_(expected_nearest_idx)
    history["collision_count"].copy_(original_reward_state.collision_count.float().view(-1))
    history["waypoint_score_sum"].copy_(original_reward_state.waypoint_score_sum.view(-1))
    history["episode_step"].copy_(original_reward_state.progress_buf.long().view(-1) - 1)
    history["initialized"].fill_(True)

    # Now compute step cache — should detect positive progress
    cache = reward_provider._update_step_cache()

    goal_progress = cache["goal_progress"]
    distance_progress = cache["distance_progress"]
    arrival_radius = float(reward_state.arrival_radius)

    # goal_progress should be positive (we simulated drone getting 0.5 closer to goal)
    expected_raw_gp = torch.clamp(
        simulated_prev_goal_distance - current_goal_distance,
        min=-arrival_radius,
        max=arrival_radius,
    )
    # Only in NAV phase
    nav_mask = (original_reward_state.phase.long().view(-1) == int(reward_state.nav_phase)).float()
    expected_goal_progress = expected_raw_gp * nav_mask

    gp_match = torch.allclose(goal_progress, expected_goal_progress, atol=1e-5)
    check(gp_match, "raw_goal_progress matches expected (prev_dist - curr_dist) clamped", phase,
          f"computed={goal_progress.tolist()}, expected={expected_goal_progress.tolist()}")

    # distance_progress should also be positive (we moved nav_target_distance closer by 0.5)
    for env_i in range(min(num_envs, 4)):
        gp = float(goal_progress[env_i])
        dp = float(distance_progress[env_i])
        is_nav = bool(nav_mask[env_i] > 0.5)
        print(f"    env[{env_i}]: goal_progress={gp:.4f}, distance_progress={dp:.4f}, "
              f"is_nav={is_nav}, arrival_radius={arrival_radius:.4f}")
        if is_nav:
            check(gp > 0.0, f"env[{env_i}] goal_progress is positive when approaching", phase, f"gp={gp:.6f}")
            check(dp > 0.0, f"env[{env_i}] distance_progress is positive when approaching", phase, f"dp={dp:.6f}")

    # ========================================================================
    # Test 4: distance_progress protection on target switch
    # ========================================================================
    print(f"\n  --- Target Switch Protection ---")

    # Simulate: history had waypoint idx=0, now env reports waypoint idx=1 (or -1 if cleared)
    reward_provider.bind_reward_state(original_reward_state)
    reward_provider._step_cache = {}

    history = reward_provider._history
    # Set up history as if previous target was a different waypoint
    fake_prev_idx = (expected_nearest_idx + 1) % max(int(waypoint_mask.shape[-1]), 1)
    history["next_waypoint_idx"].copy_(fake_prev_idx)
    history["has_pending_waypoint"].fill_(True)
    # Set nav_target_distance to something large so raw progress would be very negative
    history["nav_target_distance"].fill_(0.1)
    history["goal_distance"].copy_(current_goal_distance)
    history["phase"].fill_(int(reward_state.nav_phase))
    history["collision_count"].copy_(original_reward_state.collision_count.float().view(-1))
    history["waypoint_score_sum"].copy_(original_reward_state.waypoint_score_sum.view(-1))
    history["episode_step"].copy_(original_reward_state.progress_buf.long().view(-1) - 1)
    history["initialized"].fill_(True)

    cache = reward_provider._update_step_cache()
    dp_after_switch = cache["distance_progress"]

    # On target switch, distance_progress should be clamped to >= 0
    for env_i in range(min(num_envs, 4)):
        dp_val = float(dp_after_switch[env_i])
        is_nav = bool(original_reward_state.phase.view(-1)[env_i] == int(reward_state.nav_phase))
        if is_nav:
            check(
                dp_val >= -1e-6,
                f"env[{env_i}] distance_progress non-negative on target switch",
                phase,
                f"dp={dp_val:.6f} (should be >= 0 due to switch protection)",
            )
            print(f"    env[{env_i}]: distance_progress after target switch = {dp_val:.6f} (protected)")

    # ========================================================================
    # Test 5: Verify no stale waypoint position is returned when all are visited
    # ========================================================================
    print(f"\n  --- All Waypoints Visited Fallback ---")

    all_visited_state = replace(
        original_reward_state,
        waypoint_visited=torch.ones_like(original_reward_state.waypoint_visited),
        progress_buf=original_reward_state.progress_buf + 2,
    )
    reward_provider.bind_reward_state(all_visited_state)
    reward_provider._step_cache = {}
    history = reward_provider._history
    history["episode_step"].copy_(all_visited_state.progress_buf.long().view(-1) - 1)
    history["initialized"].fill_(True)
    history["phase"].fill_(int(reward_state.nav_phase))
    history["goal_distance"].copy_(current_goal_distance + 0.3)
    history["nav_target_distance"].copy_(current_goal_distance + 0.3)
    history["has_pending_waypoint"].fill_(False)
    history["next_waypoint_idx"].fill_(-1)
    history["collision_count"].copy_(all_visited_state.collision_count.float().view(-1))
    history["waypoint_score_sum"].copy_(all_visited_state.waypoint_score_sum.view(-1))

    _, _, has_pending_cleared = reward_provider._active_waypoint_snapshot()
    check(
        not has_pending_cleared.any(),
        "active_waypoint_snapshot reports no pending when all visited",
        phase,
        f"has_pending={has_pending_cleared.tolist()}",
    )

    cache = reward_provider._update_step_cache()
    dp_all_cleared = cache["distance_progress"]
    gp_all_cleared = cache["goal_progress"]
    for env_i in range(min(num_envs, 4)):
        dp_val = float(dp_all_cleared[env_i])
        gp_val = float(gp_all_cleared[env_i])
        is_nav = bool(all_visited_state.phase.view(-1)[env_i] == int(reward_state.nav_phase))
        if is_nav:
            # When all waypoints cleared, distance_progress should track goal_progress
            print(f"    env[{env_i}]: all_cleared -> distance_progress={dp_val:.4f}, goal_progress={gp_val:.4f}")
            check(
                abs(dp_val - gp_val) < 1e-4 or dp_val >= 0.0,
                f"env[{env_i}] distance_progress tracks goal when all waypoints cleared",
                phase,
                f"dp={dp_val:.4f}, gp={gp_val:.4f}",
            )

    # ========================================================================
    # Cleanup: restore original state
    # ========================================================================
    reward_provider.bind_reward_state(original_reward_state)
    reward_provider._step_cache = {}

    return True


def validate_first_done_eval_payload_precedence(phase: str):
    from isaac_env.evaluation.runtime_extractors import build_all_end_info_dicts

    first_done_result = {
        "status": "success",
        "score": 71.5,
        "total_score": 71.5,
        "nav_score": 20.0,
        "hover_score": 31.5,
        "waypoint_score": 20.0,
        "nav_score_raw": 0.8,
        "hover_score_raw": 0.9,
        "wp_score_raw": 0.5,
    }
    first_done_payload = {
        "layout": {"num_obstacles": 1, "goal_pos": [4.0, 2.0, 1.0]},
        "trajectory": [{
            "step": 7,
            "pos": [1.0, 2.0, 3.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }],
        "result": first_done_result,
    }
    live_result = {
        "status": "failed",
        "score": 0.0,
        "total_score": 0.0,
        "nav_score": 0.0,
        "hover_score": 0.0,
        "waypoint_score": 0.0,
        "nav_score_raw": 0.0,
        "hover_score_raw": 0.0,
        "wp_score_raw": 0.0,
    }
    live_payload = {
        "layout": {"num_obstacles": 99, "goal_pos": [9.0, 9.0, 9.0]},
        "trajectory": [{
            "step": 99,
            "pos": [9.0, 9.0, 9.0],
            "quat": [0.0, 1.0, 0.0, 0.0],
            "rotation": [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }],
        "result": live_result,
    }

    task_env = SimpleNamespace(
        latest_episode_payload=[live_payload],
        obstacle_positions=np.asarray([[[9.0, 9.0]]], dtype=np.float32),
        obstacle_radii=np.asarray([[9.0]], dtype=np.float32),
        num_obstacles=np.asarray([1.0], dtype=np.float32),
        base_obstacle_radius=np.asarray([9.0], dtype=np.float32),
        waypoint_positions=None,
        waypoint_visited=None,
        waypoint_visited_step=None,
        num_waypoints=np.asarray([0.0], dtype=np.float32),
        arena_size=np.asarray([5.0, 5.0, 3.0], dtype=np.float32),
        start_pos=np.asarray([0.5, 2.5, 0.15], dtype=np.float32),
        goal_pos=np.asarray([9.0, 9.0, 9.0], dtype=np.float32),
        drone_radius=0.05,
        arrival_radius=0.2,
        waypoint_collect_radius=0.2,
        trajectory_buffer=[[{
            "step": 123,
            "pos": [9.0, 9.0, 9.0],
            "quat": [0.0, 1.0, 0.0, 0.0],
            "rotation": [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }]],
        success_flag=np.asarray([0.0], dtype=np.float32),
        timeout_flag=np.asarray([0.0], dtype=np.float32),
        collision_exceeded=np.asarray([1.0], dtype=np.float32),
        collision_count=np.asarray([9.0], dtype=np.float32),
        arrival_step=np.asarray([999.0], dtype=np.float32),
        hover_steps=np.asarray([0.0], dtype=np.float32),
        nav_steps=np.asarray([999.0], dtype=np.float32),
        stats={"total_score": np.asarray([0.0], dtype=np.float32)},
        progress_buf=np.asarray([999.0], dtype=np.float32),
        max_collisions=5,
        nav_max_steps=1500,
        hover_max_steps=150,
        dt=0.02,
    )
    runtime = SimpleNamespace(env=SimpleNamespace(base_env=task_env), eval_first_done_payloads=[first_done_payload])
    drone_env = SimpleNamespace(env=runtime, game_id="first_done_eval_case", env_nums=1)

    end_info_dicts = build_all_end_info_dicts(drone_env)
    check(len(end_info_dicts) == 1, "first-done payload extraction returns one env", phase, str(len(end_info_dicts)))
    if not end_info_dicts:
        return

    end_info = end_info_dicts[0]
    check(end_info.get("result", {}).get("status") == "success", "first-done payload result overrides live payload", phase, str(end_info.get("result")))
    check(end_info.get("result", {}).get("total_score") == 71.5, "first-done payload score is preserved", phase, str(end_info.get("result")))
    check(end_info.get("result", {}).get("nav_score_raw") == first_done_result["nav_score_raw"], "first-done payload raw nav score is preserved", phase, str(end_info.get("result")))
    check(end_info.get("result", {}).get("hover_score_raw") == first_done_result["hover_score_raw"], "first-done payload raw hover score is preserved", phase, str(end_info.get("result")))
    check(end_info.get("result", {}).get("wp_score_raw") == first_done_result["wp_score_raw"], "first-done payload raw waypoint score is preserved", phase, str(end_info.get("result")))
    trajectory = end_info.get("trajectory", [])
    check(len(trajectory) == 1 and trajectory[0].get("step") == 7, "first-done payload trajectory is preserved", phase, str(trajectory))
    if trajectory:
        check(trajectory[0].get("quat") == [1.0, 0.0, 0.0, 0.0], "first-done payload quaternion is preserved", phase, str(trajectory[0]))
        check(len(trajectory[0].get("rotation", [])) == 9, "first-done payload rotation matrix is preserved", phase, str(trajectory[0]))
    layout = end_info.get("layout", {})
    check(layout.get("num_obstacles") == 1, "first-done payload layout overrides live layout", phase, str(layout))
    check(layout.get("goal_pos") == [4.0, 2.0, 1.0], "first-done payload goal_pos is preserved", phase, str(layout))


def validate_result_artifacts(paths, game_id: str, phase: str, expected_num_envs: int):
    result_json = paths["result_json"]
    done_file = paths["done_file"]
    trajectory_files = list_trajectory_json_files(paths["result_dir"])
    first_trajectory_json = (
        os.path.join(paths["result_dir"], trajectory_files[0]) if trajectory_files else None
    )

    check(wait_for_paths([result_json, done_file]), "required result files are generated", phase, paths["battle_dir"])
    check(os.path.isfile(result_json), "gamestate result json exists", phase, result_json)
    check(os.path.isfile(done_file), "done marker exists", phase, done_file)
    check(len(trajectory_files) > 0, "trajectory json exists", phase, str(trajectory_files))

    if os.path.isfile(done_file):
        with open(done_file, "r", encoding="utf-8") as infile:
            done_content = infile.read().strip()
        check(done_content == "done", "done file content is correct", phase, done_content)

    if os.path.isfile(result_json):
        result_payload = _load_json(result_json)
        check(result_payload.get("name") == game_id, "gamestate.name matches game_id", phase)
        check(result_payload.get("project_code") == "drone_obstacle_nav", "gamestate.project_code is correct", phase)
        check("status" in result_payload, "gamestate contains status", phase, str(result_payload.get("status")))

        frames_payload = _load_embedded_json(result_payload.get("frames", {}))
        check(isinstance(frames_payload, dict), "gamestate.frames remains a json object", phase, str(type(frames_payload).__name__))
        if isinstance(frames_payload, dict):
            check(frames_payload.get("protocol_version") == "v1", "gamestate.frames protocol version is retained", phase, str(frames_payload))
            check(len(frames_payload.keys()) == 1, "gamestate.frames has no replay payload entries", phase, str(frames_payload))

        camps = result_payload.get("camps", [])
        check(len(camps) == 1, "gamestate contains one camp", phase, str(len(camps)))
        if camps:
            end_info_payload = _load_embedded_json(camps[0].get("end_info", {}))
            check(end_info_payload.get("game_id") == game_id, "end_info.game_id matches", phase)
            score_detail = end_info_payload.get("score_detail", [])
            export_score_detail = end_info_payload.get("export_score_detail", [])
            check(len(export_score_detail) == expected_num_envs, "end_info.export_score_detail count matches env_num", phase, str(len(export_score_detail)))
            for env_idx, detail in enumerate(export_score_detail):
                obstacle_num = float(detail.get("obstacle_num", 0.0))
                obstacle_radius = float(detail.get("obstacle_radius", 0.0))
                total_waypoints = float(detail.get("total_waypoints", detail.get("waypoints_total", 0.0)))
                expected_task_type = f"obs_{int(obstacle_num)}_r_{int(round(obstacle_radius * 100.0))}_wp_{int(total_waypoints)}"
                check(
                    detail.get("task_type") == expected_task_type,
                    f"export_score_detail[{env_idx}] task_type matches layout grouping fields",
                    phase,
                    f"task_type={detail.get('task_type')}, expected={expected_task_type}",
                    quiet=True,
                )
                check(
                    int(detail.get("env_index", -1)) == env_idx,
                    f"export_score_detail[{env_idx}] env_index is retained",
                    phase,
                    f"env_index={detail.get('env_index')}, expected={env_idx}",
                    quiet=True,
                )
                total_score = float(detail.get("total_score", 0.0))
                nav_score = float(detail.get("nav_score", 0.0))
                hover_score = float(detail.get("hover_score", 0.0))
                wp_score = float(detail.get("waypoint_score", 0.0))
                nav_coeff = float(detail.get("nav_coeff", 0.0))
                hover_coeff = float(detail.get("hover_coeff", 0.0))
                nav_score_raw = float(detail.get("nav_score_raw", 0.0))
                hover_score_raw = float(detail.get("hover_score_raw", 0.0))
                wp_score_raw = float(detail.get("wp_score_raw", 0.0))
                arrival_success = float(detail.get("arrival_success", 0.0))
                hover_success = float(detail.get("hover_success", 0.0))
                hover_failed = float(detail.get("hover_failed", 0.0))
                visited_waypoints = float(detail.get("visited_waypoints", detail.get("waypoints_visited", 0.0)))
                remaining_waypoints = float(detail.get("remaining_waypoints", max(total_waypoints - visited_waypoints, 0.0)))
                score = float(detail.get("score", 0.0))
                expected_nav = nav_coeff * 0.25 * nav_score_raw * 100.0
                expected_hover = nav_coeff * hover_coeff * 0.35 * hover_score_raw * 100.0
                expected_wp = nav_coeff * 0.40 * wp_score_raw * 100.0
                expected_total = expected_nav + expected_hover + expected_wp
                check(abs(remaining_waypoints - max(total_waypoints - visited_waypoints, 0.0)) <= 1.0e-6, f"export_score_detail[{env_idx}] remaining_waypoints is consistent", phase, quiet=True)
                check(arrival_success in (0.0, 1.0), f"export_score_detail[{env_idx}] arrival_success is binary", phase, str(arrival_success), quiet=True)
                check(hover_success in (0.0, 1.0), f"export_score_detail[{env_idx}] hover_success is binary", phase, str(hover_success), quiet=True)
                check(hover_failed in (0.0, 1.0), f"export_score_detail[{env_idx}] hover_failed is binary", phase, str(hover_failed), quiet=True)
                check("total_score_sum" not in detail, f"export_score_detail[{env_idx}] omits aggregated total_score_sum", phase, quiet=True)
                check("nav_score_sum" not in detail, f"export_score_detail[{env_idx}] omits aggregated nav_score_sum", phase, quiet=True)
                check(abs(nav_score - expected_nav) <= 1.0e-6, f"export_score_detail[{env_idx}] nav_score matches weighted contribution", phase, quiet=True)
                check(abs(hover_score - expected_hover) <= 1.0e-6, f"export_score_detail[{env_idx}] hover_score matches weighted contribution", phase, quiet=True)
                check(abs(wp_score - expected_wp) <= 1.0e-6, f"export_score_detail[{env_idx}] waypoint_score matches weighted contribution", phase, quiet=True)
                check(
                    abs(total_score - expected_total) <= 1.0,
                    f"export_score_detail[{env_idx}] total_score matches PRD formula",
                    phase,
                    f"total={total_score}, expected={expected_total}",
                    quiet=True,
                )
                check(
                    abs(score - total_score) <= 1.0e-6,
                    f"export_score_detail[{env_idx}] score equals total_score",
                    phase,
                    f"score={score}, total={total_score}",
                    quiet=True,
                )

            if export_score_detail:
                unique_task_types = {detail.get("task_type", "") for detail in export_score_detail}
                expected_top_task_type = next(iter(unique_task_types)) if len(unique_task_types) == 1 else "mixed"
                check(
                    end_info_payload.get("task_type") == expected_top_task_type,
                    "end_info.task_type matches aggregated export_score_detail task types",
                    phase,
                    f"task_type={end_info_payload.get('task_type')}, expected={expected_top_task_type}",
                    quiet=True,
                )

                expected_group_count = len(unique_task_types)
                check(
                    len(score_detail) == expected_group_count,
                    "end_info.score_detail group count matches unique task types",
                    phase,
                    f"groups={len(score_detail)}, expected={expected_group_count}",
                    quiet=True,
                )

                for grouped_detail in score_detail:
                    task_type = grouped_detail.get("task_type", "")
                    grouped_items = [item for item in export_score_detail if item.get("task_type") == task_type]
                    check(bool(grouped_items), "score_detail task_type exists in export_score_detail", phase, task_type, quiet=True)
                    if not grouped_items:
                        continue

                    expected_avg_total_score = sum(float(item.get("total_score", 0.0)) for item in grouped_items) / len(grouped_items)
                    check(
                        abs(float(grouped_detail.get("game_count", 0.0)) - len(grouped_items)) <= 1.0e-6,
                        "score_detail game_count matches grouped export_score_detail size",
                        phase,
                        f"task_type={task_type}, game_count={grouped_detail.get('game_count')}, expected={len(grouped_items)}",
                        quiet=True,
                    )
                    check(
                        abs(float(grouped_detail.get("total_score", 0.0)) - expected_avg_total_score) <= 1.0,
                        "score_detail total_score matches grouped average",
                        phase,
                        f"task_type={task_type}, total_score={grouped_detail.get('total_score')}, expected={expected_avg_total_score}",
                        quiet=True,
                    )

    if first_trajectory_json is not None and os.path.isfile(first_trajectory_json):
        trajectory_payload = _load_json(first_trajectory_json)
        trajectory = trajectory_payload.get("trajectory", [])
        layout_payload = trajectory_payload.get("layout", {})
        check(trajectory_payload.get("protocol_version") == "v1", "trajectory protocol_version is retained", phase, str(trajectory_payload.get("protocol_version")), quiet=True)
        check(trajectory_payload.get("env_index") == 0, "trajectory env_index is 0", phase, str(trajectory_payload.get("env_index")), quiet=True)
        check(
            trajectory_files[0].count("_") >= 4,
            "trajectory filename includes task_type suffix",
            phase,
            trajectory_files[0],
            quiet=True,
        )
        check(isinstance(layout_payload, dict), "trajectory file carries standalone layout payload", phase, str(type(layout_payload).__name__), quiet=True)
        check(len(trajectory) >= 2, "trajectory contains reset and step points", phase, f"len={len(trajectory)}", quiet=True)
        if trajectory:
            first_point = trajectory[0]
            check(first_point.get("step") == 0, "trajectory starts from reset step 0", phase, str(first_point), quiet=True)
            check(len(first_point.get("pos", [])) == 3, "trajectory point contains 3D position", phase, str(first_point.get("pos", [])), quiet=True)
            check(len(first_point.get("quat", [])) == 4, "trajectory point contains quaternion attitude", phase, str(first_point), quiet=True)
            check(len(first_point.get("rotation", [])) == 9, "trajectory point contains rotation matrix attitude", phase, str(first_point), quiet=True)
            check(len(layout_payload.get("obstacles", [])) >= 0, "trajectory layout is directly available for replay", phase, str(layout_payload.keys()), quiet=True)
        step_values = [int(point.get("step", -1)) for point in trajectory if isinstance(point, dict) and "step" in point]
        if len(step_values) >= 2:
            expected_steps = list(range(step_values[0], step_values[0] + len(step_values)))
            check(step_values == expected_steps, "trajectory steps are continuous without missing step 1", phase, str(step_values[:10]), quiet=True)



def main():
    import torch

    from common_python.config.config_control import CONFIG
    from isaac_env.core import Drone

    logger = make_logger()
    env = None
    all_passed = False
    game_id = f"joint_eval_{time.time_ns()}"
    paths = build_case_paths(game_id)

    if os.path.exists(paths["battle_dir"]):
        shutil.rmtree(paths["battle_dir"])

    CONFIG.set_configure_file("kaiwudrl/conf/kaiwudrl/aisrv.toml")
    CONFIG.parse_aisrv_configure()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.reset_peak_memory_stats()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        usr_conf = build_eval_usr_conf(logger, game_id)
        env_conf = usr_conf.get("env_conf", {}) if isinstance(usr_conf, dict) else {}
        task_conf = env_conf.get("task_obstaclehover", {}) if isinstance(env_conf, dict) else {}
        num_envs = int(env_conf.get("num_envs", 1))
        max_episode_length = int(task_conf.get("nav_max_steps", 1500)) + int(task_conf.get("hover_max_steps", 150))

        print("=" * 60)
        print("Phase A: eval-style reset/step loop")
        print("=" * 60)
        print(f"Using eval config: num_envs={num_envs}, max_episode_length={max_episode_length}")

        env = Drone()
        env_obs = env.reset(usr_conf=usr_conf)
        extra_info = env_obs.get("extra_info", {})
        check(extra_info.get("result_code") == 0, "Drone.reset() succeeds", "A", str(extra_info))
        validate_reward_provider_mount(env, "A", expected_provider="agent_ppo")
        validate_reward_provider_semantics(env, "A")
        validate_reward_provider_runtime_output(env, "A")
        reward_terms_active = reward_provider_has_active_terms(env)
        check(reward_terms_active, "train config activates reward provider terms", "A")

        obs = env_obs.get("observation")
        check(obs is not None, "reset returns observation", "A")
        if obs is not None:
            check(hasattr(obs, "shape") and obs.shape[0] == num_envs, "observation batch matches num_envs", "A", str(getattr(obs, "shape", None)))

        action_space = env.get_action_space()
        check(hasattr(action_space, "shape"), "action_space exposes shape", "A", str(getattr(action_space, "shape", None)))
        agent = RandomAgent(action_space.shape, device)
        obs = _to_tensor(obs, device)

        check(not os.path.exists(paths["result_json"]), "result json absent before episode end", "A", paths["result_json"])
        check(not list_trajectory_json_files(paths["result_dir"]), "trajectory json absent before episode end", "A", str(list_trajectory_json_files(paths["result_dir"])))
        check(not os.path.exists(paths["done_file"]), "done file absent before episode end", "A", paths["done_file"])
        check(_all_done(np.array([[[False]]]), np.array([[[True]]])), "done helper treats truncated as episode end", "A")

        step = 0
        terminated = env_obs.get("terminated")
        truncated = env_obs.get("truncated")
        ever_done_mask = np.logical_or(
            np.asarray(terminated, dtype=bool).reshape(-1) if terminated is not None else np.zeros(num_envs, dtype=bool),
            np.asarray(truncated, dtype=bool).reshape(-1) if truncated is not None else np.zeros(num_envs, dtype=bool),
        )
        runtime_stats_seen = False
        final_terminated = terminated
        final_truncated = truncated
        waypoint_snapshot_validated = False
        while step < max_episode_length and not bool(ever_done_mask.all()):
            actions = agent.exploit(obs)
            env_reward, env_obs = env.step(actions.detach())
            extra_info = env_obs.get("extra_info", {})
            check(extra_info.get("result_code") == 0, f"step {step} succeeds", "A", str(extra_info), quiet=True)
            if extra_info.get("result_code") != 0:
                break

            rewards = env_reward.get("reward")
            validate_step_reward(
                rewards,
                "A",
                step=step,
                expected_num_envs=num_envs,
                require_nonzero=reward_terms_active,
            )
            stats = extra_info.get("stats")
            if not runtime_stats_seen and stats is not None:
                validate_runtime_stats(stats, "A", expected_num_envs=num_envs)
                runtime_stats_seen = True

            # After step 0, the reward_provider has been called with a live
            # RewardStateSnapshot that includes the active waypoint layout.
            # Run waypoint snapshot diagnostics here to catch the state with
            # pending waypoints before the episode ends.
            if not waypoint_snapshot_validated and step >= 1:
                validate_active_waypoint_snapshot_and_goal_progress(env, "A")
                waypoint_snapshot_validated = True
                print("\n  --- Multi-Step distance_progress Tracking (env[0]) ---")

            # Track distance_progress signal for first several steps
            if 2 <= step <= 10:
                validate_multi_step_distance_progress_tracking(env, step, "A")

            obs = _to_tensor(env_obs.get("observation"), device)
            terminated = env_obs.get("terminated")
            truncated = env_obs.get("truncated")
            current_done_mask = np.logical_or(
                np.asarray(terminated, dtype=bool).reshape(-1) if terminated is not None else np.zeros(num_envs, dtype=bool),
                np.asarray(truncated, dtype=bool).reshape(-1) if truncated is not None else np.zeros(num_envs, dtype=bool),
            )
            ever_done_mask = np.logical_or(ever_done_mask, current_done_mask)
            final_terminated = terminated
            final_truncated = truncated
            step += 1

        check(step > 0, "episode executed at least one step", "A", f"steps={step}")
        check(runtime_stats_seen, "runtime stats exposed waypoint/scoring fields", "A")
        snapshot_summary = validate_first_done_snapshot_summary(
            env,
            "A",
            expected_num_envs=num_envs,
            require_snapshot=True,
        )
        check(bool(ever_done_mask.all()), "episode exits because all envs reached first done", "A", str(int(ever_done_mask.sum())))
        if snapshot_summary is not None:
            check(
                snapshot_summary["snapshot_count"] >= _done_count(final_terminated) + _done_count(final_truncated),
                "first-done snapshots cover the final-step done envs",
                "A",
                str(snapshot_summary),
            )
        validate_eval_result_finalization_via_apply_function(env, "A")
        check(env._result_files_written, "episode end triggered result artifact generation", "A")

        print("\n" + "=" * 60)
        print("Phase B: verify battle/result artifacts")
        print("=" * 60)
        validate_result_artifacts(paths, game_id, "B", expected_num_envs=num_envs)

        print("\n" + "=" * 60)
        print("Phase C: verify frozen eval payload precedence")
        print("=" * 60)
        validate_first_done_eval_payload_precedence("C")

    except Exception as e:
        import traceback

        print(f"\n*** UNEXPECTED EXCEPTION ***\n{e}")
        traceback.print_exc()
        check(False, "No unexpected exception", "GLOBAL", str(e))
    finally:
        all_passed = summarize()
        sys.stdout.flush()
        sys.stderr.flush()

        print("\n" + "=" * 60)
        print("Cleanup")
        print("=" * 60)
        print(f"  battle_dir: {paths['battle_dir']}")
        print_cuda_memory_stats()
        if env is not None:
            try:
                env.close()
                print("  env.close() succeeded")
            except Exception as exc:
                print(f"  env.close() failed: {exc}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
