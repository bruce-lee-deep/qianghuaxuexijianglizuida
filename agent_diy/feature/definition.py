#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

DIY feature and sample data definitions.
DIY 特征与样本数据定义。
"""


from common_python.utils.common_func import create_cls, Frame
import torch
import numpy as np
import collections
from agent_diy.conf.conf import Config


SampleData = create_cls("SampleData", obs=None, actions=None, done=None, rewards=None)

ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls(
    "ActData",
    action=None,
)


def sample_process(collector):
    """Delegate sample processing to the collector.
    将样本处理委托给收集器。
    """
    return collector.sample_process()


def build_frame(frame_no, obs, actions, dones, rewards):
    """Build one frame of rollout data.
    构建一帧 rollout 数据。
    """

    frame = Frame(
        frame_no=frame_no,
        obs=obs,
        actions=actions,
        done=dones,
        rewards=rewards,
    )
    return frame


def obs_normalizer(obs):
    """Optional observation normalization hook.
    可选的观测归一化钩子。
    """
    pass
