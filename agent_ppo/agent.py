#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Obstacle Navigation Agent class based on kaiwudrl BaseAgent interface.
"""

import os

import numpy as np
import torch

from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.model.model import ActorCritic


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
        self.train_step = 1

        self.use_hierarchical = Config.USE_HIERARCHICAL
        self.use_rule_mpc = getattr(Config, "USE_RULE_MPC", False)
        self.hierarchical_controller = None
        if self.use_hierarchical:
            if self.use_rule_mpc:
                from agent_ppo.feature.rule_mpc_controller import RuleMPCController

                self.hierarchical_controller = RuleMPCController(logger=self.logger)
                if self.logger is not None:
                    self.logger.info("Strict linear MPC controller initialized.")
            else:
                from agent_ppo.feature.hierarchical_controller import HierarchicalController

                self.hierarchical_controller = HierarchicalController(logger=self.logger)
                if self.logger is not None:
                    self.logger.info("Hierarchical rule controller initialized.")

    def _preprocess_obs(self, obs):
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
        obs, original_shape = self._preprocess_obs(obs)

        if self.use_hierarchical and self.hierarchical_controller is not None:
            with torch.no_grad():
                actions = self.hierarchical_controller.compute_action(obs)

            for _, param in self.model.named_parameters():
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
        if self.use_hierarchical and self.hierarchical_controller is not None:
            return self._hierarchical_exploit(obs)

        obs, original_shape = self._preprocess_obs(obs)
        self.model.eval()
        with torch.no_grad():
            actions = self.model.act_inference(obs)

        (actions,) = self._reshape_output(original_shape, actions=actions)
        return actions

    def _hierarchical_exploit(self, obs):
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

        if len(original_shape) == 3:
            env_num, agent_num = original_shape[0], original_shape[1]
            actions = actions.view(env_num, agent_num, -1)
        return actions

    def learn(self, training_data):
        try:
            return self.algorithm.learn(training_data)
        except (ValueError, RuntimeError) as exc:
            msg = str(exc)
            if "nan" in msg.lower() or "invalid" in msg.lower():
                self._reset_model_weights()
                if self.logger:
                    self.logger.warning(f"learn() hit NaN/invalid state, reset model: {msg[:100]}")
                return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
            raise

    def save_model(self, path=None, id="1"):
        if not hasattr(self, "_save_path"):
            self._save_path = path if path is not None else os.path.join(os.getcwd(), "agent_ppo", "ckpt")

        save_dir = path if path is not None else self._save_path
        os.makedirs(save_dir, exist_ok=True)
        model_file_path = os.path.join(save_dir, f"model.ckpt-{str(id)}.pkl")
        torch.save(self.model.state_dict(), model_file_path)
        if self.logger:
            self.logger.info(f"saved model to {model_file_path}")

    def _reset_model_weights(self):
        self.model.init_weights()
        if self.logger:
            self.logger.warning("detected model NaN, reset weights")

    def load_model(self, path=None, id="1"):
        load_dir = path if path is not None else os.path.join(os.getcwd(), "agent_ppo", "ckpt")
        model_file_path = os.path.join(load_dir, f"model.ckpt-{str(id)}.pkl")
        if os.path.exists(model_file_path):
            self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
            if self.logger:
                self.logger.info(f"loaded model from {model_file_path}")
        else:
            raise FileNotFoundError(f"model file not found: {model_file_path}")
