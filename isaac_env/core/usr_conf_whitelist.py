#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Whitelist of user-configurable fields that may be forwarded into Isaac env.
"""


ENV_CONF_ALLOWED_FIELDS = {
    "num_envs",
    "random_seed",
}


TASK_CONF_ALLOWED_FIELDS = {
    "num_obstacles_range",
    "radius_choices",
    "nav_max_steps",
    "hover_max_steps",
    "max_collisions",
    "num_waypoints_range",
}

TRAIN_TASK_CONF_REQUIRED_FIELDS = set()

EVAL_TASK_CONF_ALLOWED_FIELDS = TASK_CONF_ALLOWED_FIELDS - TRAIN_TASK_CONF_REQUIRED_FIELDS


def select_allowed_fields(conf, allowed_fields):
    if not isinstance(conf, dict):
        return {}
    return {key: conf[key] for key in allowed_fields if key in conf}
