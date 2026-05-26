#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch

from agent_ppo.conf.conf import Config
from isaac_env.reward_provider_base import RewardProviderBase


class RewardProcess(RewardProviderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history = {}
        self._step_cache = {}

    def bind_reward_state(self, reward_state):
        previous_num_envs = self._history.get("phase").shape[0] if self._history else None
        super().bind_reward_state(reward_state)
        current_num_envs = (
            int(reward_state.num_envs) if reward_state is not None and hasattr(reward_state, "num_envs") else None
        )
        force_reset = previous_num_envs is None or current_num_envs is None or previous_num_envs != current_num_envs
        self._ensure_history(force_reset=force_reset)
        self._step_cache = {}

    def _env(self):
        if self.reward_state is None:
            raise RuntimeError("RewardProcess requires a bound env before computing rewards.")
        return self.reward_state

    @staticmethod
    def _mask_to_float(mask: torch.Tensor) -> torch.Tensor:
        return mask.float().view(-1)

    def _ensure_history(self, force_reset: bool = False):
        env = self._env()
        num_envs = int(env.num_envs)
        device = env.device

        if force_reset or not self._history or self._history["goal_distance"].shape[0] != num_envs:
            self._history = {
                "goal_distance": torch.zeros(num_envs, device=device),
                "phase": torch.full((num_envs,), -1, dtype=torch.long, device=device),
                "episode_step": torch.full((num_envs,), -1, dtype=torch.long, device=device),
                "initialized": torch.zeros(num_envs, dtype=torch.bool, device=device),
            }
            self._step_cache = {}

    def _drone_state(self):
        return self._env().drone_state

    def _drone_pos(self):
        return self._drone_state()[..., :3]

    def _current_step_id(self):
        return self._env().progress_buf.long().view(-1)

    def _prepare_step_context(self):
        env = self._env()
        self._ensure_history()

        step_id = self._current_step_id()
        current_phase = env.phase.long().view(-1)
        drone_pos = self._drone_pos()
        goal_position = env.goal_marker_positions
        goal_distance = torch.norm(goal_position - drone_pos, dim=-1).view(-1)

        history = self._history
        # Treat the first visible step or a time rollback as a fresh history baseline.
        # 将首个可见 step 或时间回退视为新的历史基线。
        reset_mask = (~history["initialized"]) | (step_id <= 1) | (step_id < history["episode_step"])
        if reset_mask.any():
            history["goal_distance"][reset_mask] = goal_distance[reset_mask]
            history["phase"][reset_mask] = current_phase[reset_mask]

        return {
            "goal_distance": goal_distance,
            "phase": current_phase,
            "step_id": step_id,
        }

    def _update_step_cache(self):
        env = self._env()
        ctx = self._prepare_step_context()
        history = self._history
        cached_step_id = self._step_cache.get("step_id")
        if cached_step_id is not None and torch.equal(cached_step_id, ctx["step_id"]):
            return self._step_cache

        reset_mask = (~history["initialized"]) | (ctx["step_id"] <= history["episode_step"])
        reset_mask = reset_mask.bool()
        nav_mask = ctx["phase"] == env.nav_phase

        raw_goal_progress = torch.clamp(
            history["goal_distance"] - ctx["goal_distance"],
            min=-env.arrival_radius,
            max=env.arrival_radius,
        )
        goal_progress = torch.where(reset_mask, torch.zeros_like(raw_goal_progress), raw_goal_progress)
        goal_progress = goal_progress * nav_mask.float()
        enter_hover = (history["phase"] == env.nav_phase) & (ctx["phase"] == env.hover_phase) & (~reset_mask)

        history["goal_distance"].copy_(ctx["goal_distance"])
        history["phase"].copy_(ctx["phase"])
        history["episode_step"].copy_(ctx["step_id"])
        history["initialized"].fill_(True)

        self._step_cache = {
            "step_id": ctx["step_id"].clone(),
            "goal_progress": goal_progress,
            "enter_hover": enter_hover.float(),
        }
        return self._step_cache

    def _reward_target_progress(self, **kwargs):
        env = self._env()
        scale = Config.REWARD_DISTANCE_SCALE
        multiplier = Config.DISTANCE_REWARD_MULTIPLIER
        target_progress = self._update_step_cache()["goal_progress"] / max(float(env.arrival_radius), 1.0e-6)
        return target_progress * scale * multiplier

    def _reward_goal_reached(self, **kwargs):
        reward_value = Config.GOAL_REACHED_REWARD
        return self._update_step_cache()["enter_hover"] * reward_value

    def _reward_in_arena(self, **kwargs):
        env = self._env()
        out_of_bounds = env._compute_out_of_bounds(self._drone_pos())
        in_arena_reward = Config.IN_ARENA_REWARD
        out_of_arena_penalty = Config.OUT_OF_ARENA_PENALTY
        return torch.where(
            out_of_bounds.view(-1),
            torch.full((env.num_envs,), -out_of_arena_penalty, device=env.device),
            torch.full((env.num_envs,), in_arena_reward, device=env.device),
        )


__all__ = ["RewardProcess"]
