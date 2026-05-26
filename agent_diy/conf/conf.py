#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO hyperparameters and model configuration for Drone Obstacle Navigation.
无人机避障导航 PPO 超参数及模型配置。
"""


class Config:
    # ========== Fixed Task Dimensions / 固定任务维度 ==========
    TASK_NAME = "ObstacleHover"
    OBS_DIM = 95
    ACTION_DIM = 4
    MAX_WAYPOINTS = 8
    OBSTACLE_FEATURE_DIM = 32
    TIME_ENCODING_DIM = 4

    # ========== Model Architecture / 模型结构 ==========
    ACTOR_HIDDEN_DIMS = [256, 128, 64]
    CRITIC_HIDDEN_DIMS = [256, 128, 64]
    ACTIVATION = "elu"
    INIT_NOISE_STD = 1.0
    FIXED_STD = False
