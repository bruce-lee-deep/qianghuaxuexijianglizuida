#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Custom algorithm interface for Drone Obstacle Navigation.
无人机避障导航自定义算法接口。
"""

import os

import torch


class Algorithm:
    def __init__(self, model, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.train_step = 0
        self.logger = logger
        self.monitor = monitor
        self._warned_no_learn = False

    def act(self, obs, critic_obs=None):
        """Generate actions, values and log-probabilities.
        生成动作、价值和对数概率。
        """
        if critic_obs is None:
            critic_obs = obs

        self.model.train()
        actions, values, log_probs = self.model(obs, critic_obs=critic_obs)

        return (
            actions,
            values,
            log_probs,
            self.model.action_mean.detach(),
            self.model.action_std.detach(),
            obs.detach(),
            critic_obs.detach(),
        )

    def learn(self, training_data):
        """Run one optimization step.
        执行一次优化步骤。
        """
        if not self._warned_no_learn and self.logger is not None:
            self.logger.warning(
                "DIY Algorithm.learn() currently returns zero losses until custom optimization is added"
            )
            self._warned_no_learn = True

        losses = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
        }
        if self.monitor:
            self.monitor.put_data({os.getpid(): losses})
        return losses
