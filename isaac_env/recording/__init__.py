#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from .camera_config import CameraConfig

__all__ = ["CameraConfig", "VideoRecorder", "MultiDroneVideoRecorder", "MultiCameraVideoWriter"]


def __getattr__(name):
    if name in {"VideoRecorder", "MultiDroneVideoRecorder"}:
        from .video_recorder import MultiDroneVideoRecorder, VideoRecorder

        return {
            "VideoRecorder": VideoRecorder,
            "MultiDroneVideoRecorder": MultiDroneVideoRecorder,
        }[name]
    if name == "MultiCameraVideoWriter":
        from .video_writer import MultiCameraVideoWriter

        return MultiCameraVideoWriter
    raise AttributeError(name)
