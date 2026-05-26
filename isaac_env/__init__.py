#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Isaac environment package exports.

Keep imports lazy so reward/unit-test code can use lightweight submodules such
as `isaac_env.reward_provider_base` without pulling the full Isaac runtime.
"""

__all__ = [
    "StandaloneDroneEnv",
    "CameraConfig",
    "VideoRecorder",
    "MultiDroneVideoRecorder",
    "MultiCameraVideoWriter",
]


def __getattr__(name):
    if name == "StandaloneDroneEnv":
        from .core import StandaloneDroneEnv

        return StandaloneDroneEnv
    if name == "CameraConfig":
        from .recording import CameraConfig

        return CameraConfig
    if name in {"VideoRecorder", "MultiDroneVideoRecorder", "MultiCameraVideoWriter"}:
        from .recording import MultiCameraVideoWriter, MultiDroneVideoRecorder, VideoRecorder

        return {
            "VideoRecorder": VideoRecorder,
            "MultiDroneVideoRecorder": MultiDroneVideoRecorder,
            "MultiCameraVideoWriter": MultiCameraVideoWriter,
        }[name]
    raise AttributeError(name)
