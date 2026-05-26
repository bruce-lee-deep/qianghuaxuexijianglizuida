#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Helpers for applying filtered user task config into Hydra task cfg.
"""


OBSTACLE_HOVER_BASE_TASK_FIELDS = (
    "num_obstacles_range",
    "radius_choices",
    "num_waypoints_range",
    "nav_max_steps",
    "hover_max_steps",
    "max_collisions",
)


def _set_cfg_value(cfg_task, field_name, value):
    if isinstance(cfg_task, dict):
        cfg_task[field_name] = value
        return
    setattr(cfg_task, field_name, value)


def apply_obstaclehover_task_conf(cfg_task, task_conf, train_task_conf_required_fields, is_eval):
    """Write filtered ObstacleHover task config into cfg.task.

    Train-only fields must be written into cfg.task during training because the
    downstream Isaac env reads them directly from task cfg. Eval must not write
    these overrides so the environment keeps its baked-in defaults.
    """

    for field_name in OBSTACLE_HOVER_BASE_TASK_FIELDS:
        _set_cfg_value(cfg_task, field_name, task_conf[field_name])

    if is_eval:
        return

    for field_name in sorted(train_task_conf_required_fields):
        _set_cfg_value(cfg_task, field_name, task_conf[field_name])
