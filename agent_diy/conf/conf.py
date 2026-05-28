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
    MODEL_SAVE_INTERVAL_SEC = 180

    # ========== Rule MPC Controller Switch ==========
    USE_RULE_MPC = True

    # ========== Observation Layout (95D) ==========
    OBS_IDX_TARGET_RPOS = [0, 1, 2]
    OBS_IDX_OBSTACLE_RPOS_START = 3
    OBS_IDX_LINEAR_VELOCITY = [27, 28, 29]
    OBS_IDX_ANGULAR_VELOCITY = [30, 31, 32]
    OBS_IDX_ROTATION_MATRIX_START = 33
    OBS_IDX_HOVER_TIMER = 42
    OBS_IDX_OBSTACLE_RADII_START = 43
    OBS_IDX_START_RPOS = [51, 52, 53]
    OBS_IDX_GOAL_RPOS = [54, 55, 56]
    OBS_IDX_PHASE_ONEHOT = [57, 58]
    OBS_IDX_WAYPOINT_RPOS_START = 59
    OBS_IDX_WAYPOINT_VISITED_START = 83
    OBS_IDX_TIME_ENCODING = [91, 92, 93, 94]
    OBSTACLE_MAX_COUNT = 8

    # ========== Physical / Timing ==========
    CTRL_DT = 0.02
    GRAVITY = 9.81
    YAW_CMD = 0.0
    MASS_NORM = 1.0

    # ========== Top-Level Target Selection ==========
    WAYPOINT_SWITCH_RADIUS = 0.22
    LOOKAHEAD_DIST_FAR = 0.60
    LOOKAHEAD_DIST_NEAR = 0.30
    WAYPOINT_HOLD_DIST = 0.20
    WAYPOINT_HOLD_SPEED = 0.08
    WAYPOINT_HOLD_FRAMES = 18
    BRAKE_TRIGGER_DIST = 0.75
    BRAKE_DECEL = 1.00
    SHORT_DIST_STRONG_BRAKE = 0.40

    # ========== MPC Settings ==========
    MPC_HORIZON = 10
    MPC_VEL_LIMIT_XY = 0.90
    MPC_VEL_LIMIT_Z = 0.45
    MPC_POS_WEIGHT = [10.0, 10.0, 16.0]
    MPC_VEL_WEIGHT = [1.5, 1.5, 2.0]
    MPC_SMOOTH_WEIGHT = [0.8, 0.8, 1.2]
    MPC_TERMINAL_POS_WEIGHT = [20.0, 20.0, 28.0]
    MPC_TERMINAL_VEL_WEIGHT = [2.0, 2.0, 3.0]
    MPC_SPEED_FAR = 0.55
    MPC_SPEED_MID = 0.35
    MPC_SPEED_NEAR = 0.18
    MPC_SPEED_CLOSE = 0.10
    TURN_SLOWDOWN_COS = 0.55
    TURN_SPEED_SCALE = 0.55

    # ========== Velocity PID ==========
    VEL_KP = [2.8, 2.8, 3.5]
    VEL_KI = [0.08, 0.08, 0.18]
    VEL_KD = [0.18, 0.18, 0.24]
    VEL_INT_LIM = [1.5, 1.5, 1.2]
    ACC_CMD_LIM = [4.0, 4.0, 5.0]

    # ========== Hover / Final Hold Position PID ==========
    HOVER_POS_KP = [1.6, 1.6, 2.2]
    HOVER_POS_KI = [0.10, 0.10, 0.24]
    HOVER_POS_KD = [0.35, 0.35, 0.40]
    HOVER_POS_INT_LIM = [0.8, 0.8, 0.6]
    HOVER_VEL_CMD_LIM = [0.35, 0.35, 0.28]

    # ========== Acceleration -> Attitude / Thrust ==========
    HOVER_THRUST_BIAS = 0.0671
    THRUST_SCALE = 0.020
    THRUST_MIN = -0.25
    THRUST_MAX = 0.25
    MAX_ROLL_CMD = 0.30
    MAX_PITCH_CMD = 0.30
    MAX_ROLL_RATE_CMD = 0.60
    MAX_PITCH_RATE_CMD = 0.60
    ROLL_P_RATE_KP = 3.2
    PITCH_P_RATE_KP = 3.2
