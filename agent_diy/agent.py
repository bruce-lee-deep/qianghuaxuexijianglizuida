#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

DIY agent entry redirected to the stable PPO implementation.
"""

from agent_ppo.agent import Agent as PPOAgent


class Agent(PPOAgent):
    """Reuse the stable PPO agent implementation for DIY config entry."""

    pass
