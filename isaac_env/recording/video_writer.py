#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Per-environment MP4 writer for recording-camera frames.
"""

from __future__ import annotations

import os

import numpy as np


class MultiCameraVideoWriter:
    """Manage one MP4 writer per environment/camera."""

    def __init__(self, video_dir: str, num_cameras: int, fps: float = 30.0, logger=None):
        self._video_dir = video_dir
        self._num_cameras = num_cameras
        self._fps = fps
        self._logger = logger
        self._writers = [None] * num_cameras
        self.recorders = self._writers
        self.is_recording = False
        self._written_frames = [0] * num_cameras

        os.makedirs(video_dir, exist_ok=True)

    def start_recording(self):
        self.stop_recording()
        self._writers = [None] * self._num_cameras
        self.recorders = self._writers
        self._written_frames = [0] * self._num_cameras
        self.is_recording = True
        if self._logger:
            self._logger.info(f"开始录制 {self._num_cameras} 路视频, fps={self._fps}, 保存目录: {self._video_dir}")

    def _normalize_frame(self, frame):
        if frame is None:
            return None
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)

        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"不支持的帧格式: shape={frame.shape}")

        if frame.dtype in (np.float32, np.float64):
            if frame.size and frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        return frame

    def _init_writer_if_needed(self, camera_idx: int, frame: np.ndarray):
        if self._writers[camera_idx] is not None:
            return

        import cv2

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path = os.path.join(self._video_dir, f"env_{camera_idx:03d}.mp4")
        writer = cv2.VideoWriter(path, fourcc, self._fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器: {path}")

        self._writers[camera_idx] = writer
        self.recorders = self._writers

        if self._logger:
            self._logger.info(f"视频写入器已初始化: env_{camera_idx:03d}.mp4, 分辨率={width}x{height}, fps={self._fps}")

    def add_frames(self, frames_list):
        if not self.is_recording:
            return

        if len(frames_list) != self._num_cameras:
            raise ValueError(f"帧数量 {len(frames_list)} 与录制路数 {self._num_cameras} 不匹配")

        import cv2

        for camera_idx, frame in enumerate(frames_list):
            if frame is None:
                continue
            frame = self._normalize_frame(frame)
            self._init_writer_if_needed(camera_idx, frame)
            self._writers[camera_idx].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self._written_frames[camera_idx] += 1

    def stop_recording(self):
        if not self.is_recording and not any(writer is not None for writer in self._writers):
            return

        for writer in self._writers:
            if writer is not None:
                writer.release()

        if self._logger:
            written_video_count = sum(1 for frame_count in self._written_frames if frame_count > 0)
            self._logger.info(f"视频写入器已关闭: {written_video_count}/{self._num_cameras} 个文件已写入")

        self._writers = [None] * self._num_cameras
        self.recorders = self._writers
        self.is_recording = False
