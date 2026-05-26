#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Arena-facing drone environment adapter.
"""


# 明确的在文件开始就设置下参数, 规避调用了其他库的函数从而不生效问题
import datetime
import os
import sys

import numpy as np
import torch
from arena_proto.arena2plat_pb2 import GameStatus
from common_python.config.config_control import CONFIG
from common_python.logging.kaiwu_logger import KaiwuLogger
from google.protobuf.timestamp_pb2 import Timestamp
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from isaac_env.evaluation import build_all_end_info_dicts, finalize_result_artifacts, validate_eval_first_done_payloads
from isaac_env.core.usr_conf_whitelist import (
    EVAL_TASK_CONF_ALLOWED_FIELDS,
    ENV_CONF_ALLOWED_FIELDS,
    TASK_CONF_ALLOWED_FIELDS,
    TRAIN_TASK_CONF_REQUIRED_FIELDS,
    select_allowed_fields,
)
from isaac_env.core.task_conf_adapter import apply_obstaclehover_task_conf

from .drone_runtime import StandaloneDroneEnv
from tools.train_env_conf_validate import extract_env_task_conf, read_usr_conf


def _resolve_random_seed():
    """Read RANDOM_SEED injected by the launcher and fail fast on invalid input.
    直接读取启动脚本注入的 RANDOM_SEED；若未注入或格式非法则立即报错。"""
    env_seed = os.environ.get("RANDOM_SEED")
    if env_seed is None:
        return int(datetime.datetime.now().timestamp() * 1000)

    try:
        return int(env_seed)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"RANDOM_SEED must be an integer, got {env_seed!r}") from exc


def _flatten_done_flags(flags):
    if flags is None:
        return np.asarray([], dtype=bool)
    if isinstance(flags, np.ndarray):
        return flags.astype(bool).reshape(-1)
    flattened = np.asarray(flags, dtype=bool)
    return flattened.reshape(-1) if flattened.size > 0 else np.asarray([], dtype=bool)


def _summarize_episode_done(terminated, truncated):
    terminated_flat = _flatten_done_flags(terminated)
    truncated_flat = _flatten_done_flags(truncated)

    if terminated_flat.size == 0 and truncated_flat.size == 0:
        done_flat = np.asarray([], dtype=bool)
    elif terminated_flat.size == 0:
        done_flat = truncated_flat.copy()
        terminated_flat = np.zeros_like(done_flat, dtype=bool)
    elif truncated_flat.size == 0:
        done_flat = terminated_flat.copy()
        truncated_flat = np.zeros_like(done_flat, dtype=bool)
    else:
        max_size = max(terminated_flat.size, truncated_flat.size)
        if terminated_flat.size != max_size:
            terminated_flat = np.resize(terminated_flat, max_size)
        if truncated_flat.size != max_size:
            truncated_flat = np.resize(truncated_flat, max_size)
        done_flat = np.logical_or(terminated_flat, truncated_flat)

    return {
        "done_flat": done_flat,
        "all_done": bool(done_flat.all()) if done_flat.size > 0 else False,
        "done_count": int(done_flat.sum()) if done_flat.size > 0 else 0,
        "terminated_count": int(terminated_flat.sum()) if terminated_flat.size > 0 else 0,
        "truncated_count": int(truncated_flat.sum()) if truncated_flat.size > 0 else 0,
        "env_count": int(done_flat.size),
    }


class Drone:
    """无人机环境。"""

    def __init__(self):
        self.min_episode_length = 50
        self.max_episode_length = 500
        self.frame_no = 0

        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.set_logger_format(
            f"/{CONFIG.svr_name}/kaiwu_env_pid{self.current_pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log"
        )
        self.logger.info(f"kaiwu_env start at pid {self.current_pid}, ")

        self.env = None
        self.eval_write_json_file = False
        self.capture_interval = 5
        self.enable_video = False
        self.video_dir = None
        self._result_files_written = False

    def _create_env(self, cfg: DictConfig):
        if self.env is None:
            if self.enable_video:
                self.video_dir = f"/workspace/battle/{self.game_id}/mp4"
                os.makedirs(self.video_dir, exist_ok=True)

            self.env = StandaloneDroneEnv(
                cfg,
                enable_video_recording=self.enable_video,
                video_save_path=self.video_dir,
                eval_mode=self.is_eval,
                logger=self.logger,
            )

        self.sim_device = cfg.sim.device if hasattr(cfg.sim, "device") else "cuda"
        self.frame_no = 1
        self.env_nums = self.env.get_num_envs()
        self.num_actions = self.env.get_action_space().shape[-1] if hasattr(self.env.get_action_space(), "shape") else 12
        obs_space = self.env.get_observation_space()
        self.num_obs = obs_space.shape[-1] if hasattr(obs_space, "shape") else self.num_actions
        if self.enable_video:
            self.env.create_camera()

    def reset(self, usr_conf):
        self.game_id = usr_conf["game_id"]
        self.is_eval = usr_conf["is_eval"]
        self._result_files_written = False
        self.usr_conf = usr_conf

        try:
            env_conf, task_name, _, task_conf = extract_env_task_conf(usr_conf)
            env_conf = select_allowed_fields(env_conf, ENV_CONF_ALLOWED_FIELDS)
            allowed_task_fields = EVAL_TASK_CONF_ALLOWED_FIELDS if self.is_eval else TASK_CONF_ALLOWED_FIELDS
            task_conf = select_allowed_fields(task_conf, allowed_task_fields)

            config_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "OmniDrones-2026-kaiwu", "cfg")
            )
            compose_overrides = [f"task={task_name}"] if task_name else []
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="train", overrides=compose_overrides)
                OmegaConf.set_struct(cfg, False)

            # 训练允许 env_conf.random_seed 覆盖平台环境变量；评估固定使用平台注入的 RANDOM_SEED。
            platform_seed = _resolve_random_seed()
            cfg.seed = int(env_conf.get("random_seed", platform_seed))
            cfg.task.name = task_name
            if "num_envs" not in env_conf:
                raise KeyError("Missing required config field: env_conf.num_envs")
            cfg.task.env.num_envs = env_conf["num_envs"]

            if cfg.task.name == "ObstacleHover":
                required_task_fields = (
                    "num_obstacles_range",
                    "radius_choices",
                    "num_waypoints_range",
                    "nav_max_steps",
                    "hover_max_steps",
                    "max_collisions",
                )
                if not self.is_eval:
                    required_task_fields = required_task_fields + tuple(sorted(TRAIN_TASK_CONF_REQUIRED_FIELDS))
                missing_fields = [field_name for field_name in required_task_fields if field_name not in task_conf]
                if missing_fields:
                    missing_field_names = ", ".join(
                        f"env_conf.task_obstaclehover.{field_name}" for field_name in missing_fields
                    )
                    raise KeyError(f"Missing required task config field(s): {missing_field_names}")

                apply_obstaclehover_task_conf(cfg.task, task_conf, TRAIN_TASK_CONF_REQUIRED_FIELDS, is_eval=self.is_eval)
                cfg.task.record_trajectory = bool(task_conf.get("record_trajectory", False))
                if self.is_eval:
                    cfg.task.record_trajectory = True
                cfg.task.usr_conf = OmegaConf.create(usr_conf)
                cfg.task.env.max_episode_length = int(cfg.task.nav_max_steps) + int(cfg.task.hover_max_steps)

            if hasattr(cfg, "env") and cfg.env is not None:
                cfg.env.num_envs = cfg.task.env.num_envs
                cfg.env.min_episode_length = cfg.task.env.min_episode_length
                cfg.env.max_episode_length = cfg.task.env.max_episode_length
                cfg.env.env_spacing = cfg.task.env.env_spacing

            self.enable_video = False
            self.cfg = cfg
            self.min_episode_length = self.cfg.env.min_episode_length
            self.max_episode_length = self.cfg.env.max_episode_length
            self._create_env(self.cfg)

            if self.is_eval:
                self.battle_dir = f"/workspace/battle/{self.game_id}"
                if not os.path.exists(self.battle_dir):
                    os.makedirs(self.battle_dir)
            now = datetime.datetime.utcnow()
            self.start_timestamp = Timestamp()
            self.start_timestamp.FromDatetime(now)

            self.env.clear_cached_episode_payloads()
            reset_state, terminated, truncated = self.env.reset()
            self.frame_no = 1
            obs = self.env._extract_observation(reset_state) if hasattr(self.env, "_extract_observation") else None

            extra_info = {"result_code": 0, "result_message": "OK"}
            return {
                "env_id": self.game_id,
                "frame_no": self.frame_no,
                "observation": obs,
                "extra_info": extra_info,
                "terminated": terminated,
                "truncated": truncated,
            }

        except Exception as e:
            self.logger.exception("reset error")
            extra_info = {"result_code": -1, "result_message": str(e)}
            return {
                "env_id": self.game_id,
                "frame_no": self.frame_no,
                "observation": {},
                "extra_info": extra_info,
                "terminated": False,
                "truncated": False,
            }

    def get_action_space(self):
        try:
            return self.env.get_action_space()
        except Exception:
            self.logger.exception("get_action_space error")
            return None

    def get_observation_space(self):
        try:
            return self.env.get_observation_space()
        except Exception:
            self.logger.exception("get_observation_space error")
            return None

    def step(self, actions):

        result_code = 0
        result_message = "OK"

        try:
            self.frame_no += 1
            _, obs, rewards, terminated, truncated, stats = self.env.step(actions)

            extra_info = {"result_code": result_code, "result_message": result_message}
            if stats is not None:
                extra_info["stats"] = stats
            env_reward = {"env_id": self.game_id, "frame_no": self.frame_no, "reward": rewards}
            env_obs = {
                "env_id": self.game_id,
                "frame_no": self.frame_no,
                "observation": obs,
                "extra_info": extra_info,
                "terminated": terminated,
                "truncated": truncated,
            }

            done_summary = _summarize_episode_done(terminated, truncated)
            all_done = done_summary["all_done"]
            env_count = done_summary["env_count"]
            done_count = done_summary["done_count"]
            terminated_count = done_summary["terminated_count"]
            truncated_count = done_summary["truncated_count"]

            reached_max_length = self.frame_no > self.max_episode_length
            if self.is_eval and (all_done or reached_max_length or self.frame_no % 100 == 0):
                self.logger.info(
                    f"[终止检查] frame_no={self.frame_no}/{self.max_episode_length}, is_eval={self.is_eval}, all_done={all_done}, reached_max_length={reached_max_length}, done_count={done_count}/{env_count}, terminated_count={terminated_count}, truncated_count={truncated_count}"
                )

            if self.is_eval and (all_done or reached_max_length):
                if all_done:
                    trigger_reason = "所有环境已结束(done=terminated|truncated)"
                else:
                    trigger_reason = "达到最大长度"
                self.logger.info(
                    f"[Episode终止] 触发条件: {trigger_reason}, frame_no={self.frame_no}, max_length={self.max_episode_length}"
                )
                self.logger.info("[Episode终止] 正在生成结果文件...")
                try:
                    self.make_json_and_done_file()
                    print("[Episode终止] 结果文件已生成，Episode完成")
                except Exception as e:
                    self.logger.error(f"[Episode终止] 生成结果文件失败: {e}")
                    raise

            return env_reward, env_obs

        except Exception as e:
            self.logger.exception("_handle_step error")
            extra_info = {"result_code": -2, "result_message": str(e)}
            zero_reward = np.zeros((self.env_nums, 1, 1), dtype=np.float32) if hasattr(self, "env_nums") else np.zeros(1, dtype=np.float32)
            obs_dim = getattr(self, "num_obs", self.num_actions)
            zero_obs = np.zeros((self.env_nums, 1, obs_dim), dtype=np.float32) if hasattr(self, "env_nums") else np.zeros(1, dtype=np.float32)
            env_reward = {"env_id": self.game_id, "frame_no": self.frame_no, "reward": zero_reward}
            env_obs = {
                "env_id": self.game_id,
                "frame_no": self.frame_no,
                "observation": zero_obs,
                "extra_info": extra_info,
                "terminated": np.zeros((self.env_nums, 1, 1), dtype=bool) if hasattr(self, "env_nums") else False,
                "truncated": np.zeros((self.env_nums, 1, 1), dtype=bool) if hasattr(self, "env_nums") else False,
            }
            return env_reward, env_obs

    def make_json_and_done_file(self):
        if self._result_files_written:
            return

        if self.is_eval:
            snapshot_summary = validate_eval_first_done_payloads(self, expected_env_count=self.env_nums)
            snapshot_count = snapshot_summary["snapshot_count"]
            missing_envs = snapshot_summary["missing_envs"]
            self.logger.info(f"[结果生成] first-done snapshot coverage: {snapshot_count}/{self.env_nums}")
            if missing_envs:
                raise RuntimeError(f"评估结果生成失败: missing first-done snapshots for envs={missing_envs}")

        end_info_dicts = build_all_end_info_dicts(self)
        finalize_result_artifacts(self, end_info_dicts)
        self._result_files_written = True

    def get_eval_first_done_snapshot_summary(self, expected_env_count=None):
        runtime = getattr(self, "env", None)
        first_done_payloads = getattr(runtime, "eval_first_done_payloads", None)
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

    def close(self):
        if self.env is not None:
            if hasattr(self.env, "close"):
                self.env.close()
            self.env = None

        if torch.cuda.is_available():
            for _ in range(3):
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError as e:
        if "already been set" in str(e):
            print(f"[INFO] {str(e)}")

    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "eval"]:
        print("ERROR: 必须指定运行模式 (train/eval)")
        print("正确用法: python -m isaac_env.core.arena_env [train|eval]")
        sys.exit(1)

    valid_modes = ["train", "eval"]
    mode = None
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg in valid_modes and mode is None:
            mode = arg
            continue
        new_argv.append(arg)

    if not mode:
        print("ERROR: 必须指定运行模式 (train/eval)")
        print("正确用法: python -m isaac_env.core.arena_env [train|eval]")
        sys.exit(1)

    sys.argv = new_argv
    game_id = "1"

    configure_file = "kaiwudrl/conf/kaiwudrl/aisrv.toml"
    CONFIG.set_configure_file(configure_file)
    CONFIG.parse_aisrv_configure()

    import logging
    _logger = logging.getLogger("arena_env.__main__")
    _logger.setLevel(logging.INFO)
    if not _logger.handlers:
        _logger.addHandler(logging.StreamHandler())

    if mode == "eval":
        usr_conf = read_usr_conf("tools/eval/conf/eval_env_conf.toml", _logger, eval=True)
        usr_conf["game_id"] = game_id
        usr_conf["is_eval"] = True
    else:
        usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", _logger, eval=False)
        usr_conf["game_id"] = game_id
        usr_conf["is_eval"] = False

    env = Drone()
    try:
        data = env.reset(usr_conf)
        extra_info = data["extra_info"]
        if extra_info["result_code"] != 0:
            print("env.reset failed")
        else:
            obs = data.get("observation")
            if obs is not None and hasattr(obs, "shape"):
                print(f"Reset obs shape: {obs.shape}")
            else:
                print("Reset obs is None or has no 'shape' attribute")

            for i in range(300):
                action_space = env.get_action_space()
                if hasattr(action_space, "shape"):
                    act_shape = action_space.shape
                elif isinstance(action_space, (list, tuple)) and len(action_space) > 0 and hasattr(action_space[0], "shape"):
                    act_shape = action_space[0].shape
                else:
                    raise RuntimeError(f"Unsupported action_space type for sampling: {type(action_space)}")
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                random_tensor = torch.randn(act_shape, device=device)

                env_reward, env_obs = env.step(random_tensor)
                extra_info = env_obs["extra_info"]
                if extra_info["result_code"] != 0:
                    print(f"Step {i} failed: {extra_info['result_message']}")
                    break
                rewards = env_reward.get("reward")
                if rewards is not None and hasattr(rewards, "mean"):
                    print(f"Step {i}: rewards mean = {rewards.mean().item():.4f}")
                else:
                    print(f"Step {i}: completed")

    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        print("Test finished")
        env.close()


if __name__ == "__main__":
    import multiprocessing as mp

    main()
