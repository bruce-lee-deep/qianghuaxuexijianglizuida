#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO training workflow for Drone Obstacle Navigation.
无人机避障导航 PPO 训练工作流。
"""

import time
from collections import deque

import os
import torch
import numpy as np
from agent_ppo.feature.definition import ReplayBuffer
from agent_ppo.feature.reward_process import RewardProcess
from agent_ppo.conf.conf import Config
from tools.cluster_monitor import ClusterMonitorTracker
from tools.train_env_conf_validate import read_usr_conf, extract_env_task_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """PPO training workflow entry point.
    PPO 训练工作流入口。
    """
    env = envs[0]
    agent = agents[0]

    episode_runner = PPOTrainer(
        env=env,
        agent=agent,
        logger=logger,
        monitor=monitor,
    )

    episode_runner.train()


class PPOTrainer:
    """PPO training loop: handles rollout, learning, logging and checkpointing.
    PPO 训练循环：处理数据采集、学习更新、日志记录和模型保存。
    """

    _TASK_METRIC_KEYS = (
        "success",
        "timeout",
        "failed",
        "collision_exceeded",
        "arrival_success",
        "hover_success",
        "hover_failed",
        "total_score",
        "nav_coeff",
        "hover_coeff",
        "nav_score",
        "hover_score",
        "waypoint_score",
        "nav_score_raw",
        "hover_score_raw",
        "wp_score_raw",
        "time_norm",
        "smooth_norm",
        "waypoints_visited",
        "waypoints_total",
        "remaining_waypoints",
        "collision_count",
        "max_collisions",
        "hover_precision",
    )
    _TASK_ENV_STAT_KEYS = {
        "success": "success",
        "timeout": "timeout",
        "collision_exceeded": "collision_exceeded",
        "arrival_success": "arrival_success",
        "hover_success": "hover_success",
        "hover_failed": "hover_failed",
        "total_score": "total_score",
        "nav_coeff": "nav_coeff",
        "hover_coeff": "hover_coeff",
        "nav_score": "nav_score",
        "hover_score": "hover_score",
        "waypoint_score": "waypoint_score",
        "nav_score_raw": "nav_score_raw",
        "hover_score_raw": "hover_score_raw",
        "wp_score_raw": "wp_score_raw",
        "time_norm": "time_norm",
        "smooth_norm": "smooth_norm",
        "waypoints_visited": "waypoints_visited",
        "waypoints_total": "waypoints_total",
        "remaining_waypoints": "remaining_waypoints",
        "collision_count": "collision_count",
        "max_collisions": "max_collisions",
        "hover_precision": "hover_precision",
    }
    _TASK_LOG_FIELDS = (
        ("task_success", "Success", ".3f"),
        ("task_timeout", "Timeout", ".3f"),
        ("task_failed", "OOBFail", ".3f"),
        ("task_collision_exceeded", "CollisionFail", ".3f"),
        ("task_total_score", "Score", ".2f"),
        ("task_arrival_success", "ArrivalRate", ".3f"),
        ("task_hover_success", "HoverSucc", ".3f"),
        ("task_hover_failed", "HoverFail", ".3f"),
        ("task_nav_coeff", "NavOK", ".3f"),
        ("task_hover_coeff", "HoverOK", ".3f"),
        ("task_nav_score_raw", "NavRaw", ".3f"),
        ("task_hover_score_raw", "HoverRaw", ".3f"),
        ("task_wp_score_raw", "WPRaw", ".3f"),
        ("task_waypoint_score", "WPScore", ".3f"),
        ("task_waypoints_visited", "WPVisited", ".3f"),
        ("task_waypoints_total", "WPTotal", ".3f"),
        ("task_collision_count", "CollisionCnt", ".3f"),
        ("task_max_collisions", "CollisionCap", ".3f"),
        ("task_hover_precision", "HoverPrecision", ".3f"),
    )
    _TASK_MONITOR_FIELDS = {
        "task_success": "success",
        "task_timeout": "timeout",
        "task_failed": "failed",
        "task_collision_exceeded": "collision_exceeded",
        "task_total_score": "total_score",
        "task_nav_score": "nav_score",
        "task_hover_score": "hover_score",
        "task_waypoint_score": "waypoint_score",
        "task_nav_score_raw": "nav_score_raw",
        "task_hover_score_raw": "hover_score_raw",
        "task_wp_score_raw": "wp_score_raw",
        "task_time_norm": "time_norm",
        "task_smooth_norm": "smooth_norm",
        "task_waypoints_visited": "waypoints_visited",
        "task_waypoints_total": "waypoints_total",
        "task_collision_count": "collision_count",
        "task_max_collisions": "max_collisions",
        "task_hover_precision": "hover_precision",
    }
    _REQUIRED_ENV_STAT_KEYS = tuple(sorted(set(_TASK_ENV_STAT_KEYS.values()) | {"success", "timeout"}))
    _MONITOR_REPORT_INTERVAL_SEC = 60.0

    @staticmethod
    def _is_train_test_mode():
        """Return whether the current run is a train_test smoke test.
        判断当前运行是否为 train_test 冒烟测试。
        """
        return os.getenv("is_train_test", "").strip().lower() in {"1", "true", "yes", "on"}

    def __init__(self, env, agent, logger, monitor):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.monitor = monitor
        self.iteration_cnt = 0
        self.total_frames = 0

        self._current_obs = None
        self._env_initialized = False

        self._stats_window_size = 100
        self._episode_returns = deque(maxlen=self._stats_window_size)
        self._episode_lengths = deque(maxlen=self._stats_window_size)
        self._current_episode_return = None
        self._current_episode_length = None
        self._total_completed_episodes = 0
        self._reported_completed_episodes = 0
        self._last_monitor_report_time = 0.0

        # Task-specific metrics collected when an episode terminates.
        # 在 episode 结束时收集的任务指标。
        self._task_metrics_history = {k: deque(maxlen=self._stats_window_size) for k in self._TASK_METRIC_KEYS}
        self._latest_stats = None
        self._env_stats_schema_checked = False
        self._cluster_monitor = ClusterMonitorTracker(window_size=self._stats_window_size)
        self._env_error_count = 0

        self._usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", self.logger)
        env_conf, task_name, task_key, task_conf = extract_env_task_conf(self._usr_conf)
        if self._is_train_test_mode():
            train_test_num_envs = 10
            env_conf["num_envs"] = train_test_num_envs
            if isinstance(self._usr_conf.get("env_conf"), dict):
                self._usr_conf["env_conf"]["num_envs"] = train_test_num_envs
            if hasattr(self.agent, "num_envs"):
                self.agent.num_envs = train_test_num_envs
            if self.logger is not None:
                self.logger.info(
                    "检测到 train_test 模式，运行时将 env_conf.num_envs 覆盖为 %s",
                    train_test_num_envs,
                )
        if str(task_name).lower() != str(Config.TASK_NAME).lower():
            raise ValueError(f"Unsupported task_name '{task_name}', only '{Config.TASK_NAME}' is supported")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_envs = env_conf["num_envs"]
        self.obs_dim = Config.OBS_DIM
        self.action_dim = Config.ACTION_DIM
        self.num_steps = max(int(getattr(Config, "NUM_STEPS_PER_ENV", 64)), 1)
        self.frames_per_batch = self.num_envs * self.num_steps
        self._current_episode_return = torch.zeros(self.num_envs, device=self.device)
        self._current_episode_length = torch.zeros(self.num_envs, device=self.device)

        # Instantiate a local RewardProcess only for reward-term discovery and logging.
        # 本地 RewardProcess 仅用于奖励项发现和日志展示；实际奖励计算仍由环境侧挂载执行。
        reward_configs = self._parse_reward_configs(self._usr_conf)
        self.reward_process = RewardProcess(
            reward_configs=reward_configs,
            task_config={},
            num_envs=self.num_envs,
            device=self.device,
        )

    @staticmethod
    def _parse_reward_configs(usr_conf):
        """Parse [rewards.*] sections from user configuration.
        从用户配置中解析 [rewards.*] 段。

        Args:
            usr_conf: Parsed TOML configuration dict.

        Returns:
            Dict of {term_name: {"weight": float, "params": dict}}.
        """
        rewards_section = usr_conf.get("rewards", {})
        if not isinstance(rewards_section, dict):
            return {}
        configs = {}
        for term_name, term_cfg in rewards_section.items():
            if not isinstance(term_cfg, dict):
                continue
            configs[term_name] = {
                "weight": float(term_cfg.get("weight", 0.0)),
                "params": dict(term_cfg.get("params", {})),
            }
        return configs

    @classmethod
    def _round_monitor_value(cls, key, value):
        """Apply monitor-friendly precision.
        对监控值做精度控制。
        """
        if not isinstance(value, (float, np.floating)):
            return value

        return round(float(value), 2)

    def _to_tensor(self, data, expected_shape, dtype=torch.float32, name="data"):
        """Convert data to CUDA tensor with fallback to zeros for unexpected types.
        将数据转换为 CUDA tensor，对异常类型返回零值张量兜底。
        """
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype).to(self.device)
        if isinstance(data, torch.Tensor):
            return data.to(dtype).to(self.device)
        self._env_error_count += 1
        if self._env_error_count <= 10:
            self.logger.warning(f"Unexpected {name} type: {type(data).__name__}, using zeros{list(expected_shape)}")
        return torch.zeros(expected_shape, dtype=dtype, device=self.device)

    def _ensure_env_initialized(self):
        """Initialize the environment on first call.
        首次调用时初始化环境。
        """
        if not self._env_initialized:
            try:
                env_obs = self.env.reset(usr_conf=self._usr_conf)
                if not isinstance(env_obs, dict):
                    raise TypeError(f"env.reset() must return dict, got {type(env_obs).__name__}")

                extra_info = env_obs.get("extra_info")
                if isinstance(extra_info, dict) and extra_info.get("result_code", 0) != 0:
                    result_message = extra_info.get("result_message", "")
                    raise RuntimeError(f"环境 reset 失败: {result_message}")

                obs = env_obs["observation"]
                obs = self._to_tensor(obs, (self.num_envs, 1, self.obs_dim), name="reset_obs")
                self._current_obs = obs
                self._env_initialized = True

                self.logger.info("环境初始化完成")
            except Exception:
                self.logger.exception("环境 reset 异常")
                raise

    def _validate_env_stats_schema(self, env_stats):
        """Validate the fixed obstacle_hover stats schema once.
        校验 obstacle_hover 固定 stats schema。
        """
        if not isinstance(env_stats, dict):
            raise TypeError(f"extra_info['stats'] must be dict, got {type(env_stats).__name__}")

        missing_keys = [key for key in self._REQUIRED_ENV_STAT_KEYS if key not in env_stats]
        if missing_keys:
            raise KeyError(f"Missing required env stats keys: {missing_keys}")

        expected_shape = (self.num_envs, 1)
        for key in self._REQUIRED_ENV_STAT_KEYS:
            value = env_stats[key]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"env stats[{key!r}] must be numpy.ndarray, got {type(value).__name__}")
            if value.shape != expected_shape:
                raise ValueError(f"env stats[{key!r}] shape mismatch: expected {expected_shape}, got {value.shape}")

    def _cache_env_stats(self, env_stats):
        """Cache the latest stats batch from the environment.
        缓存环境最新一批 stats。
        """
        if env_stats is None:
            raise KeyError("extra_info['stats'] is required for obstacle_hover training")
        if not self._env_stats_schema_checked:
            self._validate_env_stats_schema(env_stats)
            self._env_stats_schema_checked = True
        self._latest_stats = env_stats

    def _build_done_task_metrics(self, env_idx):
        """Build one finished episode metric row from fixed env stats.
        从固定环境 stats 中构建单个完成 episode 的指标。
        """
        task_metrics = {
            metric_key: float(self._latest_stats[stat_key][env_idx, 0])
            for metric_key, stat_key in self._TASK_ENV_STAT_KEYS.items()
        }
        task_metrics["failed"] = float(
            task_metrics["success"] < 0.5
            and task_metrics["timeout"] < 0.5
            and task_metrics["collision_exceeded"] < 0.5
            and task_metrics["hover_failed"] < 0.5
        )
        return task_metrics

    def _build_task_log_parts(self, stats):
        """Format aggregated task metrics for logging.
        将聚合任务指标格式化为日志字段。
        """
        if stats["completed_episodes"] <= 0:
            return []

        task_msg_parts = [f"{label}={stats[stat_key]:{fmt}}" for stat_key, label, fmt in self._TASK_LOG_FIELDS]
        task_msg_parts.append(f"NavSub=({stats['task_time_norm']:.3f},{stats['task_smooth_norm']:.3f})")
        task_msg_parts.append(f"WP={stats['task_waypoints_visited']:.1f}/{stats['task_waypoints_total']:.1f}")
        return task_msg_parts

    def _build_stats_snapshot(self, rollout_return=None, completed_episodes=0):
        """Build the latest aggregated stats snapshot for logging/monitoring.
        构建当前最新的聚合统计快照，用于日志和监控。
        """
        stats = {
            "completed_episodes": int(completed_episodes),
            "total_completed_episodes": int(self._total_completed_episodes),
            "total_frames": int(self.total_frames),
            "env_error_count": int(self._env_error_count),
        }
        if rollout_return is not None:
            stats["rollout_return"] = float(rollout_return)

        if self._episode_returns:
            stats["mean_episode_return"] = np.mean(list(self._episode_returns))
            stats["mean_episode_length"] = np.mean(list(self._episode_lengths))

        for key in self._TASK_METRIC_KEYS:
            history = self._task_metrics_history[key]
            if history:
                stats[f"task_{key}"] = np.mean(list(history))
        return stats

    def _build_monitor_data(self, stats, episode_cnt=None):
        """Convert rollout stats into monitor payload.
        将 rollout 统计转换为监控上报载荷。
        """
        if episode_cnt is None:
            episode_cnt = max(self._total_completed_episodes, 0)

        monitor_data = {
            # episode_cnt reports the cumulative number of completed env episodes.
            # episode_cnt 表示自本次训练流程启动以来累计完成的 env 局数。
            "episode_cnt": int(episode_cnt),
        }

        if "mean_episode_return" in stats:
            monitor_data["reward"] = stats["mean_episode_return"]
        elif "rollout_return" in stats:
            monitor_data["reward"] = stats["rollout_return"]

        if "mean_episode_length" in stats:
            monitor_data["step"] = stats["mean_episode_length"]

        has_task_summary = stats.get("total_completed_episodes", 0) > 0 and all(
            key in stats for key in ("task_arrival_success", "task_success")
        )
        if has_task_summary:
            arrival_rate = stats["task_arrival_success"]
            monitor_data["arrival_rate"] = arrival_rate

            for stat_key, monitor_key in self._TASK_MONITOR_FIELDS.items():
                monitor_data[monitor_key] = stats[stat_key]

        monitor_data.update(self._cluster_monitor.build_monitor_data())
        return {key: self._round_monitor_value(key, value) for key, value in monitor_data.items()}

    def _has_monitor_signal(self, stats):
        """Return whether stats already contain monitor-worthy rollout/task/reward data.
        判断当前 stats 是否已经包含足以上报监控的 rollout/任务/奖励信息。
        """
        if not isinstance(stats, dict):
            return False

        if "mean_episode_return" in stats or "rollout_return" in stats:
            return True

        return False

    def _maybe_report_monitor(self, stats=None, force=False):
        """Report environment/task monitor metrics when the report interval elapses.
        在达到上报周期时及时推送环境/任务监控指标。
        """
        now = time.time()
        if not force and (now - self._last_monitor_report_time) < self._MONITOR_REPORT_INTERVAL_SEC:
            return False

        if stats is None:
            stats = self._build_stats_snapshot()

        episode_cnt_total = max(self._total_completed_episodes, 0)
        if not force and episode_cnt_total <= 0 and not self._has_monitor_signal(stats):
            return False

        monitor_data = self._build_monitor_data(stats, episode_cnt=episode_cnt_total)
        self._last_monitor_report_time = now
        self._reported_completed_episodes = self._total_completed_episodes

        if self.monitor:
            self.monitor.put_data({os.getpid(): monitor_data})
        return True

    def rollout(self):
        """Collect num_steps of experience data for training.
        采集 num_steps 步经验数据用于训练。
        """
        env = self.env
        agent = self.agent

        replay_buffer = ReplayBuffer(
            num_envs=self.num_envs,
            num_steps=self.num_steps,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )
        replay_buffer.clear()

        self._ensure_env_initialized()
        obs = self._current_obs

        completed_episodes = 0
        for step in range(self.num_steps):
            policy_obs, _ = agent._preprocess_obs(obs)
            with torch.no_grad():
                actions, values, log_probs = agent.predict(obs)

            try:
                env_reward, env_obs = env.step(actions.detach())
            except Exception:
                self.logger.exception(f"环境 step 异常 (step={step})")
                raise

            rewards = env_reward["reward"]
            next_obs = env_obs["observation"]
            terminated = env_obs["terminated"]
            truncated = env_obs["truncated"]
            dones = np.logical_or(terminated, truncated).astype(np.float32)
            extra_info = env_obs["extra_info"]

            env_stats = extra_info.get("stats") if isinstance(extra_info, dict) else None
            self._cache_env_stats(env_stats)

            # Convert environment outputs to tensors.
            # 将环境输出转换为张量。
            next_obs = self._to_tensor(next_obs, (self.num_envs, 1, self.obs_dim), name="obs")
            rewards = self._to_tensor(rewards, (self.num_envs, 1, 1), name="rewards")
            dones = self._to_tensor(dones, (self.num_envs, 1, 1), name="dones")

            obs_flat = policy_obs.view(self.num_envs, self.obs_dim)
            actions_flat = actions.view(self.num_envs, self.action_dim)
            values_flat = values.view(self.num_envs)
            log_probs_flat = log_probs.view(self.num_envs)
            rewards_flat = rewards.view(self.num_envs)
            dones_flat = dones.view(self.num_envs)

            replay_buffer.add(
                obs=obs_flat,
                actions=actions_flat,
                rewards=rewards_flat,
                values=values_flat,
                log_probs=log_probs_flat,
                dones=dones_flat,
            )

            # Update episode-level statistics.
            # 更新 episode 级统计信息。
            self._current_episode_return += rewards_flat
            self._current_episode_length += 1

            # Handle environments that have completed an episode.
            # 处理已经完成一个 episode 的环境实例。
            done_mask = dones_flat > 0.5
            if done_mask.any():
                done_indices = done_mask.nonzero(as_tuple=True)[0]
                for idx in done_indices:
                    self._episode_returns.append(self._current_episode_return[idx].item())
                    self._episode_lengths.append(self._current_episode_length[idx].item())
                    completed_episodes += 1
                    self._total_completed_episodes += 1

                    task_metrics = self._build_done_task_metrics(idx.item())
                    for key, value in task_metrics.items():
                        self._task_metrics_history[key].append(value)

                self._cluster_monitor.add_from_stats_batch(
                    self._latest_stats,
                    done_indices.detach().cpu().tolist(),
                )

                # Reset statistics for completed environments.
                # 重置已完成环境的统计信息。
                self._current_episode_return[done_mask] = 0
                self._current_episode_length[done_mask] = 0
                if getattr(agent, "hierarchical_controller", None) is not None:
                    agent.hierarchical_controller.reset_envs(done_mask)

            obs = next_obs
            self.total_frames += self.num_envs
            self._maybe_report_monitor()

        # Keep the latest observation for the next rollout.
        # 保存最新观测供下一次 rollout 使用。
        self._current_obs = obs

        rollout_return = replay_buffer.rewards.sum(dim=0).mean().item()

        # Bootstrap the value estimate of the last step.
        # 计算最后一步的 bootstrap 价值估计。
        with torch.no_grad():
            _, next_values, _ = agent.predict(obs)
        next_values_flat = next_values.view(self.num_envs)

        # Prepare flattened training data, including GAE.
        # 准备展平后的训练数据，并计算 GAE。
        try:
            training_data = self.agent.algorithm.prepare_training_data(replay_buffer, next_values_flat)
            if training_data is not None:
                stats = self._build_stats_snapshot(
                    rollout_return=rollout_return,
                    completed_episodes=completed_episodes,
                )
                yield training_data, stats
        except ValueError as e:
            if "Buffer is empty" in str(e):
                self.logger.info(f"Iter {self.iteration_cnt}: Buffer is empty, skipping training")
            else:
                raise e

    def train(self):
        """Main training loop: rollout → learn → log → checkpoint.
        主训练循环：采集数据 → 学习更新 → 日志记录 → 保存模型。
        """
        last_save_agent_time = time.time()
        self._last_monitor_report_time = time.time()

        self.logger.info(f"使用设备: {self.device}")
        self.logger.info(f"环境数量: {self.num_envs}")
        self.logger.info(f"观测维度: {self.obs_dim}, 动作维度: {self.action_dim}")
        self.logger.info(f"每次训练步数 (train_every): {self.num_steps}")
        self.logger.info(f"每次训练帧数 (frames_per_batch): {self.frames_per_batch}")
        self.logger.info("=" * 60)
        self.logger.info("开始训练...")

        while True:
            self.iteration_cnt += 1

            episode_start_time = time.time()

            # 模型定期保存（纯规则模式同样生效）
            now = time.time()
            if now - last_save_agent_time >= Config.MODEL_SAVE_INTERVAL_SEC:
                self.logger.info(f"开始保存模型 (距上次 {now - last_save_agent_time:.0f}s)")
                self.agent.save_model()
                last_save_agent_time = now

            rollout_start = time.time()
            for training_data, stats in self.rollout():
                rollout_time = time.time() - rollout_start
                fps = self.frames_per_batch / max(rollout_time, 1e-6)

                log_msg = (
                    f"Iter {self.iteration_cnt} Rollout: "
                    f"Time={rollout_time:.2f}s, "
                    f"FPS={fps:.0f}, "
                    f"Frames={stats['total_frames']:,}, "
                    f"Return={stats['rollout_return']:.2f}"
                )
                if "mean_episode_return" in stats:
                    log_msg += f", EpReturn={stats['mean_episode_return']:.2f}"
                    log_msg += f", EpLen={stats['mean_episode_length']:.1f}"
                log_msg += f", CompletedEps={stats['completed_episodes']}"
                if stats["env_error_count"] > 0:
                    log_msg += f", EnvErrors={stats['env_error_count']}"
                self.logger.info(log_msg)

                task_msg_parts = self._build_task_log_parts(stats)
                if task_msg_parts:
                    self.logger.info(f"Iter {self.iteration_cnt} Task: {', '.join(task_msg_parts)}")

                # Run one learning update.
                learn_start = time.time()
                losses = self.agent.learn(training_data)
                learn_time = time.time() - learn_start
                self.logger.info(
                    f"Iter {self.iteration_cnt} Learn: "
                    f"PolicyLoss={losses['policy_loss']:6.3f}, "
                    f"ValueLoss={losses['value_loss']:6.3f}, "
                    f"Entropy={losses['entropy_loss']:6.3f}, "
                    f"Time={learn_time:.2f}s"
                )

                self._maybe_report_monitor(stats=stats)
