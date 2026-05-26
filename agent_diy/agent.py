#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Obstacle Navigation custom agent.
无人机避障导航自定义智能体。
"""

import os

import torch
import numpy as np

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwudrl.interface.agent import BaseAgent
from agent_diy.model.model import Model
from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(agent_type, device, logger, monitor)

        self.cur_model_name = ""
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.obs_dim = Config.OBS_DIM
        self.action_dim = Config.ACTION_DIM

        self.model = Model(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            actor_hidden_dims=Config.ACTOR_HIDDEN_DIMS,
            critic_hidden_dims=Config.CRITIC_HIDDEN_DIMS,
            activation=Config.ACTIVATION,
            init_noise_std=Config.INIT_NOISE_STD,
            fixed_std=Config.FIXED_STD,
        ).to(self.device)
        self.algorithm = Algorithm(model=self.model, device=self.device, logger=self.logger, monitor=self.monitor)

    def _split_obs(self, obs):
        """Normalize optional actor/critic observation inputs.
        统一处理可选的 actor/critic 观测输入。
        """
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs
        return obs, obs

    def _preprocess_obs(self, obs):
        """Convert observations to tensors and flatten batched agent dims.
        将观测转换为张量，并展平多智能体批维度。
        """
        if isinstance(obs, tuple):
            obs = obs[0]

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        elif isinstance(obs, torch.Tensor):
            obs = obs.to(self.device, dtype=torch.float32)
        else:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        original_shape = tuple(obs.shape)
        if obs.dim() == 3:
            obs = obs.view(obs.shape[0] * obs.shape[1], -1)

        return obs, original_shape

    def _reshape_output(self, original_shape, **tensors):
        """Restore flattened outputs to the original observation layout.
        将展平后的输出恢复为原始观测布局。
        """
        if len(original_shape) == 3:
            env_num, agent_num = original_shape[0], original_shape[1]
            result = {}
            for name, tensor in tensors.items():
                if tensor.dim() == 1:
                    result[name] = tensor.view(env_num, agent_num)
                else:
                    result[name] = tensor.view(env_num, agent_num, -1)
            return tuple(result.values())
        return tuple(tensors.values())

    def predict(self, obs):
        """Sample actions for training.
        为训练阶段采样动作。
        """
        actor_obs, critic_obs = self._split_obs(obs)
        actor_obs, original_shape = self._preprocess_obs(actor_obs)
        critic_obs, _ = self._preprocess_obs(critic_obs)

        with torch.no_grad():
            actions, values, log_probs, _, _, _, _ = self.algorithm.act(actor_obs, critic_obs)

        return self._reshape_output(original_shape, actions=actions, values=values, log_probs=log_probs)

    def exploit(self, obs):
        """Produce deterministic actions for evaluation.
        为评估阶段生成确定性动作。
        """
        actor_obs, _ = self._split_obs(obs)
        actor_obs, original_shape = self._preprocess_obs(actor_obs)

        self.model.eval()
        with torch.no_grad():
            actions = self.model.act_inference(actor_obs)

        (actions,) = self._reshape_output(original_shape, actions=actions)
        return actions

    def learn(self, training_data):
        """Run one learning step.
        执行一次学习步骤。
        """
        return self.algorithm.learn(training_data)

    def predict_local(self, obs, critic_obs=None):
        """Run local inference with optional critic observations.
        使用可选的 critic 观测执行本地推理。
        """
        actor_obs, _ = self._preprocess_obs(obs)
        if critic_obs is None:
            critic_obs = actor_obs
        else:
            critic_obs, _ = self._preprocess_obs(critic_obs)
        return self.algorithm.act(actor_obs, critic_obs)

    def action_process(self, act_data):
        """Hook for custom action post-processing.
        自定义动作后处理钩子。
        """
        return act_data

    def observation_process(self, obs_q):
        """Hook for custom observation pre-processing.
        自定义观测预处理钩子。
        """
        return obs_q

    def reset(self):
        """Reset agent-side transient state.
        重置智能体侧的临时状态。
        """
        return None

    def save_model(self, path=None, id="1"):
        """Save model parameters.
        保存模型参数。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"saved model to {model_file_path}")

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
