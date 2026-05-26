#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Drone Obstacle Navigation.
无人机避障导航 PPO 算法实现。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(
        self,
        model,
        device="cuda",
        logger=None,
        monitor=None,
    ):
        self.model = model
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.gamma = Config.GAMMA
        self.lam = Config.LAM
        self.clip_param = Config.CLIP_PARAM
        self.value_loss_coef = Config.VALUE_LOSS_COEF
        self.entropy_coef = Config.ENTROPY_COEF
        self.max_grad_norm = Config.MAX_GRAD_NORM
        self.num_epochs = Config.NUM_LEARNING_EPOCHS
        self.num_minibatches = Config.NUM_MINI_BATCHES

        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

    def compute_gae(self, rewards, values, dones, next_values):
        """Compute Generalized Advantage Estimation (GAE).
        计算广义优势估计（GAE）。

        Args:
            rewards: [num_steps, num_envs] reward sequence.
            values: [num_steps, num_envs] value estimates.
            dones: [num_steps, num_envs] episode termination flags.
            next_values: [num_envs] value estimate of the last step.

        Returns:
            returns: [num_steps, num_envs] discounted returns.
            advantages: [num_steps, num_envs] advantage estimates.
        """
        num_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_non_terminal = 1.0 - dones[step].float()
                next_value = next_values
            else:
                next_non_terminal = 1.0 - dones[step].float()
                next_value = values[step + 1]

            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = gae + values[step]

        return returns, advantages

    def prepare_training_data(self, replay_buffer, next_values):
        """Prepare flattened training data from ReplayBuffer with GAE computation.
        从 ReplayBuffer 准备展平的训练数据（含 GAE 计算）。

        Args:
            replay_buffer: ReplayBuffer instance.
            next_values: [num_envs] value estimate of the last step.

        Returns:
            Tuple of (obs, actions, log_probs, returns, advantages), all flattened.
        """
        observations, actions, rewards, values, log_probs, dones = replay_buffer.get()
        returns, advantages = self.compute_gae(rewards, values, dones, next_values)

        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        obs_flat = observations.view(-1, obs_dim)
        actions_flat = actions.view(-1, action_dim)
        log_probs_flat = log_probs.view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)

        return obs_flat, actions_flat, log_probs_flat, returns_flat, advantages_flat

    def learn(self, training_data):
        """PPO learning update with clipped surrogate objective.
        PPO 学习更新（裁剪替代目标函数）。

        Args:
            training_data: Tuple of (obs, actions, old_log_probs, returns, advantages).

        Returns:
            Dictionary of loss metrics and PPO diagnostics.
            损失指标及 PPO 诊断信息字典。
        """
        obs, actions, old_log_probs, returns, advantages = training_data

        total_samples = obs.shape[0]
        batch_size = total_samples // self.num_minibatches

        # Normalize advantages
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        sum_policy_loss = 0.0
        sum_value_loss = 0.0
        sum_entropy = 0.0
        sum_total_loss = 0.0
        sum_approx_kl = 0.0
        sum_clip_fraction = 0.0
        num_updates = 0

        for epoch in range(self.num_epochs):
            indices = torch.randperm(total_samples, device=obs.device)

            for i in range(self.num_minibatches):
                start = i * batch_size
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass
                # 前向传播
                self.model.train()
                self.model.update_distribution(batch_obs)
                new_log_probs = self.model.get_actions_log_prob(batch_actions)
                values = self.model.evaluate(batch_obs)
                entropy = self.model.entropy.mean()

                # Policy loss (PPO clip)
                # 策略损失（PPO 裁剪）
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                # 价值损失
                value_loss = (batch_returns - values).pow(2).mean()

                # Total loss
                # 总损失
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Backward pass
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # PPO diagnostic metrics
                # PPO 诊断指标
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()

                sum_policy_loss += policy_loss.item()
                sum_value_loss += value_loss.item()
                sum_entropy += entropy.item()
                sum_total_loss += loss.item()
                sum_approx_kl += approx_kl
                sum_clip_fraction += clip_fraction
                num_updates += 1

        n = max(num_updates, 1)
        avg_policy_loss = sum_policy_loss / n
        avg_value_loss = sum_value_loss / n
        avg_entropy = sum_entropy / n
        avg_total_loss = sum_total_loss / n
        avg_approx_kl = sum_approx_kl / n
        avg_clip_fraction = sum_clip_fraction / n

        # Explained variance: measures how well the value function fits the returns
        # Explained variance：衡量价值函数对回报的拟合程度
        with torch.no_grad():
            self.model.eval()
            all_values = self.model.evaluate(obs)
            var_returns = returns.var()
            explained_var = (1 - (returns - all_values).var() / (var_returns + 1e-8)).item()

        current_lr = self.optimizer.param_groups[0]["lr"]

        avg_losses = {
            "policy_loss": round(avg_policy_loss, 2),
            "value_loss": round(avg_value_loss, 2),
            "entropy_loss": round(avg_entropy, 2),
            "total_loss": round(avg_total_loss, 2),
        }

        if self.monitor:
            self.monitor.put_data({os.getpid(): avg_losses})
        return avg_losses
