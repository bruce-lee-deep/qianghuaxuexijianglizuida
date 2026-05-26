#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
import torch


def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def safe_float(value, default=0.0):
    try:
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return float(default)
            value = value.reshape(-1)[0]
        if torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            value = value.detach().cpu().reshape(-1)[0].item()
        if isinstance(value, np.generic):
            value = value.item()
        return float(value)
    except Exception as exc:
        raise ValueError(f"failed to coerce value to float: {value!r}") from exc


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"failed to coerce value to int: {value!r}") from exc


def ensure_float_list(values):
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        values = [values]
    return [safe_float(item) for item in values]


def safe_mean(values):
    valid_values = [float(value) for value in values if value is not None]
    if not valid_values:
        return 0.0
    return sum(valid_values) / len(valid_values)
