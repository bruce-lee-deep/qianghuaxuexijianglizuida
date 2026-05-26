#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""DIY reward wrapper for ObstacleHover.
ObstacleHover 的 DIY 奖励包装器。

The runtime entry is kept under the DIY namespace so downstream code can import
the expected symbol without duplicating reward formulas here.
保留 DIY 命名空间下的运行时入口，便于下游按既有路径导入，同时避免在此重复维护奖励公式。
"""

from agent_diy.feature.reward_process import RewardProcess as PPORewardProcess


class RewardProcess(PPORewardProcess):
    pass


__all__ = ["RewardProcess"]
