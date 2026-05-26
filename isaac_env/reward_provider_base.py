#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Reward provider base for ObstacleHover.

The reward bridge remains configuration-driven, but concrete reward formulas
read from a bound read-only reward state snapshot instead of a live env.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch


class RewardProviderBase:
    REWARD_PREFIX = "_reward_"

    def __init__(self, reward_configs=None, task_config=None, num_envs=1, device="cuda", env=None):
        self.reward_configs = reward_configs or {}
        self.task_config = task_config or {}
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        # Compatibility note: historical callers passed `env=` here.
        # The provider now only stores a read-only reward-state snapshot.
        self.reward_state = env

        self._terms = {}
        # Weighted per-term reward contributions from the latest compute_reward() call.
        # 最近一次 compute_reward() 中各奖励子项的“带权贡献值”，用于监控奖励组成。
        self._last_term_values = {}
        for attr_name in dir(self):
            if not attr_name.startswith(self.REWARD_PREFIX):
                continue
            method = getattr(self, attr_name, None)
            if not callable(method):
                continue
            term_name = attr_name[len(self.REWARD_PREFIX) :]
            cfg = self.reward_configs.get(term_name, {})
            self._terms[term_name] = {
                "method": method,
                "weight": float(cfg.get("weight", 0.0)),
                "params": dict(cfg.get("params", {})),
            }

    def bind_reward_state(self, reward_state):
        self.reward_state = reward_state

    def _current_num_envs(self) -> int:
        if self.reward_state is not None and hasattr(self.reward_state, "num_envs"):
            return int(self.reward_state.num_envs)
        return self.num_envs

    def _current_device(self) -> torch.device:
        if self.reward_state is not None and hasattr(self.reward_state, "device"):
            return torch.device(self.reward_state.device)
        return self.device

    def _to_tensor(self, value: Any) -> torch.Tensor:
        device = self._current_device()
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=device, dtype=torch.float32)
        if np.isscalar(value):
            return torch.full((self._current_num_envs(),), float(value), device=device, dtype=torch.float32)
        return torch.as_tensor(value, device=device, dtype=torch.float32)

    def _zeros(self) -> torch.Tensor:
        return torch.zeros(self._current_num_envs(), device=self._current_device(), dtype=torch.float32)

    def _call_reward_method(self, method, **kwargs):
        sig = inspect.signature(method)
        accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
        if accepts_var_kw:
            return method(**kwargs)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return method(**filtered_kwargs)

    def compute_reward(self, env_reward=None, reward_state=None, **kwargs):
        if reward_state is not None:
            self.bind_reward_state(reward_state)
        if self.reward_state is None:
            raise RuntimeError(
                "RewardProviderBase.compute_reward() requires a bound reward state. "
                "Pass reward_state=... or call bind_reward_state() first."
            )

        active_terms = [info for info in self._terms.values() if abs(info["weight"]) > 1.0e-8]
        self._last_term_values = {}
        if not active_terms:
            if env_reward is not None:
                return self._to_tensor(env_reward).view(self._current_num_envs())
            return self._zeros()

        reward = self._zeros()
        for term_name, term_info in self._terms.items():
            weight = term_info["weight"]
            if abs(weight) < 1.0e-8:
                continue

            term_value = self._call_reward_method(
                term_info["method"],
                env_reward=env_reward,
                **term_info["params"],
                **kwargs,
            )
            weighted_value = weight * self._to_tensor(term_value).view(self._current_num_envs())
            self._last_term_values[term_name] = weighted_value.detach()
            reward = reward + weighted_value
        return reward


__all__ = ["RewardProviderBase"]
