#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

ReplayBuffer for PPO on-policy training data collection.
PPO 在线策略训练数据收集用的经验回放缓冲区。
"""


import torch


class ReplayBuffer:
    """Replay buffer for PPO on-policy training.
    PPO 在线策略训练用经验回放缓冲区，负责数据的存储、获取和清空。
    """

    def __init__(self, num_envs, num_steps, obs_dim, action_dim, device="cpu"):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.observations = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)

        self.step = 0
        self.is_full = False

    def add(self, obs, actions, rewards, values, log_probs, dones):
        """Add one step of experience data.
        添加一步经验数据。

        Args:
            obs: [num_envs, obs_dim].
            actions: [num_envs, action_dim].
            rewards: [num_envs].
            values: [num_envs].
            log_probs: [num_envs].
            dones: [num_envs].
        """
        if self.step >= self.num_steps:
            self.is_full = True
            self.step = 0

        self.observations[self.step].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.values[self.step].copy_(values)
        self.log_probs[self.step].copy_(log_probs)
        self.dones[self.step].copy_(dones)

        self.step += 1

    def get(self):
        """Get all stored data from the buffer.
        获取缓冲区中的所有数据。

        Returns:
            Tuple of (observations, actions, rewards, values, log_probs, dones).
        """
        if self.step == 0:
            raise ValueError("Buffer is empty")

        actual_steps = self.step if not self.is_full else self.num_steps

        return (
            self.observations[:actual_steps],
            self.actions[:actual_steps],
            self.rewards[:actual_steps],
            self.values[:actual_steps],
            self.log_probs[:actual_steps],
            self.dones[:actual_steps],
        )

    def clear(self):
        """Clear the buffer.
        清空缓冲区。
        """
        self.step = 0
        self.is_full = False
        self.observations.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.dones.zero_()
