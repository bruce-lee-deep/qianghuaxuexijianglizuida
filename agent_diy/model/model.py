#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Custom policy/value model for Drone Obstacle Navigation.
无人机避障导航自定义策略/价值模型。
"""


import torch
from torch import nn
from torch.distributions import Normal

from agent_diy.conf.conf import Config


class Model(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        fixed_std=False,
    ):
        super(Model, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        actor_hidden_dims = actor_hidden_dims or Config.ACTOR_HIDDEN_DIMS
        critic_hidden_dims = critic_hidden_dims or Config.CRITIC_HIDDEN_DIMS
        activation_fn = get_activation(activation)

        self.actor = self._build_mlp(obs_dim, action_dim, actor_hidden_dims, activation_fn)
        self.critic = self._build_mlp(obs_dim, 1, critic_hidden_dims, activation_fn)

        std = init_noise_std * torch.ones(action_dim)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None

        Normal.set_default_validate_args = False
        self.init_weights()

    @staticmethod
    def _build_mlp(input_dim, output_dim, hidden_dims, activation_fn):
        """Build a simple feed-forward network.
        构建简单的前馈网络。
        """
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(type(activation_fn)())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize linear layers with orthogonal weights.
        使用正交权重初始化线性层。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, observations, critic_obs=None):
        """Sample actions and evaluate values.
        采样动作并评估价值。
        """
        if critic_obs is None:
            critic_obs = observations

        self.update_distribution(observations)
        actions = self.distribution.sample()
        log_probs = self.distribution.log_prob(actions).sum(dim=-1)
        values = self.evaluate(critic_obs)
        return actions, values, log_probs

    @property
    def action_mean(self):
        return self.distribution.mean if self.distribution is not None else None

    @property
    def action_std(self):
        return self.distribution.stddev if self.distribution is not None else None

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1) if self.distribution is not None else None

    def update_distribution(self, observations):
        """Update the action distribution from observations.
        根据观测更新动作分布。
        """
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, std)

    def act(self, observations, deterministic=False):
        """Generate actions from observations.
        根据观测生成动作。
        """
        self.update_distribution(observations)
        if deterministic:
            return self.distribution.mean
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Compute log-probabilities of given actions.
        计算给定动作的对数概率。
        """
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call update_distribution first.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, observations):
        """Evaluate state values.
        评估状态价值。
        """
        values = self.critic(observations)
        return values.squeeze(-1)

    def act_inference(self, observations):
        """Generate deterministic inference actions.
        生成推理阶段的确定性动作。
        """
        with torch.no_grad():
            return self.actor(observations)


def get_activation(act_name):
    """Resolve activation function by name.
    根据名称解析激活函数。
    """
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "crelu":
        return nn.ReLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    return nn.ReLU()
