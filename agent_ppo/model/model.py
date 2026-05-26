#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Actor-Critic network for continuous control in Drone Obstacle Navigation.
无人机避障导航连续动作空间 Actor-Critic 网络。

Usage / 使用方式:
    model = ActorCritic(obs_dim, action_dim)
    actions, values, log_probs = model(observations)
"""

import torch

torch.set_num_interop_threads(1)
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from agent_ppo.conf.conf import Config


class ActorCritic(nn.Module):
    """Actor-Critic model with separate MLP networks and Gaussian action distribution.
    独立 MLP 网络 + 高斯动作分布的 Actor-Critic 模型。
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
        fixed_std=False,
    ):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        activation_fn = get_activation(activation)

        # Build the actor network.
        # 构建策略网络。
        actor_layers = []
        input_dim = obs_dim
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(activation_fn)
            input_dim = hidden_dim
        actor_layers.append(nn.Linear(input_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Build the critic network.
        # 构建价值网络。
        critic_layers = []
        input_dim = obs_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(activation_fn)
            input_dim = hidden_dim
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        std = init_noise_std * torch.ones(action_dim)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None

        Normal.set_default_validate_args = False
        self.init_weights()

    def init_weights(self):
        """Initialize weights with orthogonal initialization.
        使用正交初始化网络权重。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, observations, history=None, masks=None, hidden_states=None):
        """Forward pass: sample actions and compute values.
        前向传播：采样动作并计算价值。

        Args:
            observations: [batch_size, obs_dim].

        Returns:
            actions: [batch_size, action_dim].
            values: [batch_size].
            log_probs: [batch_size].
        """
        if self.distribution is not None:
            del self.distribution
            self.distribution = None

        action_mean = self.actor(observations)
        values = self.critic(observations)

        std = self.std.to(action_mean.device)
        self.distribution = Normal(action_mean, std)
        actions = self.distribution.sample()
        log_probs = self.distribution.log_prob(actions).sum(dim=-1)

        return actions, values.squeeze(-1), log_probs

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
        """Update the action distribution given observations.
        根据观测更新动作分布。
        """
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, std)

    def act(self, observations, deterministic=False):
        """Generate actions from observations.
        根据观测生成动作。

        Args:
            observations: [batch_size, obs_dim].
            deterministic: If True, return mean actions.

        Returns:
            actions: [batch_size, action_dim].
        """
        self.update_distribution(observations)

        if deterministic:
            return self.distribution.mean
        else:
            return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Compute log probability of given actions.
        计算给定动作的对数概率。
        """
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call update_distribution first.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, observations):
        """Evaluate the value of observations.
        评估观测的价值。

        Args:
            observations: [batch_size, obs_dim].

        Returns:
            values: [batch_size].
        """
        values = self.critic(observations)
        return values.squeeze(-1)

    def act_inference(self, observations):
        """Deterministic action generation for inference/evaluation.
        推理/评估时的确定性动作生成。

        Args:
            observations: [batch_size, obs_dim].

        Returns:
            actions: [batch_size, action_dim].
        """
        with torch.no_grad():
            action_mean = self.actor(observations)
        return action_mean

    def train(self, mode=True):
        super().train(mode)
        return self

    def test(self):
        return self.eval()


def get_activation(act_name):
    """Get activation function by name.
    根据名称获取激活函数。
    """
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print(f"Invalid activation function: {act_name}, using ReLU instead")
        return nn.ReLU()
