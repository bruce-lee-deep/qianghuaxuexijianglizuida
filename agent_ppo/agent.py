#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Obstacle Navigation Agent class based on kaiwudrl BaseAgent interface.
无人机避障导航 Agent 主类，基于 kaiwudrl BaseAgent 接口。
"""

import os
import torch
import numpy as np
from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.model.model import ActorCritic
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.obs_dim = Config.OBS_DIM
        self.action_dim = Config.ACTION_DIM

        self.model = ActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            actor_hidden_dims=Config.ACTOR_HIDDEN_DIMS,
            critic_hidden_dims=Config.CRITIC_HIDDEN_DIMS,
            activation=Config.ACTIVATION,
            init_noise_std=Config.INIT_NOISE_STD,
            fixed_std=Config.FIXED_STD,
        ).to(device)

        self.algorithm = Algorithm(model=self.model, device=device, logger=logger, monitor=monitor)

        # 纯规则模式下框架不更新 train_step，设为 1 确保能保存模型
        self.train_step = 1

        # Hierarchical controller for rule-based evaluation.
        # Only created when USE_HIERARCHICAL is True; training still uses the
        # neural network via predict().
        self.use_hierarchical = Config.USE_HIERARCHICAL
        self.hierarchical_controller = None
        if self.use_hierarchical:
            from agent_ppo.feature.hierarchical_controller import HierarchicalController

            self.hierarchical_controller = HierarchicalController(logger=self.logger)
            if self.logger is not None:
                self.logger.info("分层控制器已初始化，eval 模式将使用规则驱动控制")

    def _preprocess_obs(self, obs):
        """Unified observation preprocessing: type conversion + dimension flattening.
        统一的观测预处理：类型转换 + 维度展平。

        Args:
            obs: Raw observation, supports tuple/np.ndarray/torch.Tensor,
                 shape [env_num, agent_num, obs_dim] or [batch_size, obs_dim].
                 原始观测，支持 tuple/np.ndarray/torch.Tensor，
                 维度为 [env_num, agent_num, obs_dim] 或 [batch_size, obs_dim]。

        Returns:
            obs_flat: [batch_size, obs_dim] CUDA Tensor.
            original_shape: Original shape for restoring output dimensions.
        """
        if isinstance(obs, tuple):
            obs = obs[0]

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        elif obs.device != torch.device(self.device):
            obs = obs.to(self.device)

        original_shape = obs.shape

        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Unexpected observation dim {obs.shape[-1]}, expected {self.obs_dim}")

        if obs.dim() == 3:
            obs = obs.view(obs.shape[0] * obs.shape[1], -1)

        return obs, original_shape

    def _reshape_output(self, original_shape, **tensors):
        """Restore output dimensions according to the original input shape.
        根据原始输入 shape 恢复输出维度。

        Args:
            original_shape: Original shape returned by _preprocess_obs.
            **tensors: Named tensors to reshape (e.g. actions, values, log_probs).

        Returns:
            Tuple of reshaped tensors in the same order as input.
        """
        if len(original_shape) == 3:
            env_num, agent_num = original_shape[0], original_shape[1]
            result = {}
            for name, t in tensors.items():
                if t.dim() == 1:
                    result[name] = t.view(env_num, agent_num)
                else:
                    result[name] = t.view(env_num, agent_num, -1)
            return tuple(result.values())
        return tuple(tensors.values())

    def predict(self, obs):
        """Predict actions in training mode (stochastic).
        训练模式下预测动作（随机采样）。

        当 USE_HIERARCHICAL 时，动作由分层控制器生成，
        但 values 和 log_probs 仍从网络计算，保证 RL 训练正常进行。

        Args:
            obs: [env_num, agent_num, obs_dim] or [batch_size, obs_dim].
        Returns:
            actions, values, log_probs.
        """
        obs, original_shape = self._preprocess_obs(obs)

        if self.use_hierarchical and self.hierarchical_controller is not None:
            with torch.no_grad():
                actions = self.hierarchical_controller.compute_action(obs)
            # 先检查模型参数是否 NaN，有问题就重置
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    self._reset_model_weights()
                    break
            try:
                self.model.train()
                self.model.update_distribution(obs)
                values = self.model.critic(obs).squeeze(-1)
                log_probs = self.model.get_actions_log_prob(actions)
                if torch.isnan(values).any() or torch.isnan(log_probs).any():
                    self._reset_model_weights()
                    values = torch.zeros(obs.shape[0], device=obs.device)
                    log_probs = torch.zeros(obs.shape[0], device=obs.device)
            except (ValueError, RuntimeError):
                self._reset_model_weights()
                values = torch.zeros(obs.shape[0], device=obs.device)
                log_probs = torch.zeros(obs.shape[0], device=obs.device)
            return self._reshape_output(original_shape, actions=actions, values=values, log_probs=log_probs)

        self.model.train()
        actions, values, log_probs = self.model(obs)

        return self._reshape_output(original_shape, actions=actions, values=values, log_probs=log_probs)

    def exploit(self, obs):
        """Exploit mode (deterministic actions).
        利用模式（确定性动作）。

        When USE_HIERARCHICAL is enabled, delegates to the rule-based
        hierarchical controller instead of the learned policy.

        Args:
            obs: [env_num, agent_num, obs_dim] or [batch_size, obs_dim].
        Returns:
            actions.
        """
        if self.use_hierarchical and self.hierarchical_controller is not None:
            return self._hierarchical_exploit(obs)

        obs, original_shape = self._preprocess_obs(obs)

        self.model.eval()
        with torch.no_grad():
            actions = self.model.act_inference(obs)

        (actions,) = self._reshape_output(original_shape, actions=actions)
        return actions

    def _hierarchical_exploit(self, obs):
        """Rule-based hierarchical control exploit path.

        The hierarchical controller parses the observation, plans a path,
        and computes control actions — all without a learned policy.
        """
        if isinstance(obs, tuple):
            obs = obs[0]

        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
        elif obs.device != torch.device(self.device):
            obs_tensor = obs.to(self.device)
        else:
            obs_tensor = obs

        original_shape = obs_tensor.shape

        if obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.view(obs_tensor.shape[0] * obs_tensor.shape[1], -1)

        with torch.no_grad():
            actions = self.hierarchical_controller.compute_action(obs_tensor)

        # Restore original shape.
        if len(original_shape) == 3:
            env_num, agent_num = original_shape[0], original_shape[1]
            actions = actions.view(env_num, agent_num, -1)

        return actions

    def learn(self, training_data):
        """Run one learning update.
        执行一次学习更新。

        Args:
            training_data: Tuple of (obs, actions, old_log_probs, returns, advantages).
        Returns:
            Loss dictionary.
        """
        try:
            return self.algorithm.learn(training_data)
        except (ValueError, RuntimeError) as e:
            msg = str(e)
            if "nan" in msg.lower() or "invalid" in msg.lower():
                self._reset_model_weights()
                if self.logger:
                    self.logger.warning(f"learn() 遇到 NaN，已重置模型: {msg[:100]}")
                return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
            raise

    def save_model(self, path=None, id="1"):
        """Save model parameters.
        保存模型参数。
        """
        import os as _os
        # 记住框架第一次调用时的正确路径，后续沿用
        if not hasattr(self, "_save_path"):
            if path is not None:
                self._save_path = path
            else:
                self._save_path = _os.path.join(_os.getcwd(), "agent_ppo", "ckpt")
        save_dir = path if path is not None else self._save_path
        _os.makedirs(save_dir, exist_ok=True)
        model_file_path = _os.path.join(save_dir, f"model.ckpt-{str(id)}.pkl")
        try:
            torch.save(self.model.state_dict(), model_file_path)
            self.logger.info(f"模型已保存: {model_file_path}")
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")

    def _reset_model_weights(self):
        """重置模型权重到初始状态（NaN 恢复）。"""
        self.model.init_weights()
        if self.logger:
            self.logger.warning("检测到模型 NaN，已重置权重")

    def load_model(self, path=None, id="1"):
        """Load model parameters if the checkpoint exists.
        如果 checkpoint 存在则加载模型参数。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if os.path.exists(model_file_path):
            self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
            self.logger.info(f"loaded model from {model_file_path}")
        else:
            raise FileNotFoundError(f"model file not found: {model_file_path}")
