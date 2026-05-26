#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import cv2
import numpy as np


class VideoRecorder:
    """视频录制器, 支持在headless模式下录制视频"""

    def __init__(
        self,
        save_path: str,
        fps: int = 30,
        resolution: tuple = (1920, 1080),
        logger=None,
    ):
        self.save_path = save_path
        self.logger = logger
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.frame_count = 0
        self.is_recording = False

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def start_recording(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.frame_count = 0
        self.is_recording = True
        self.logger.info(f"开始录制视频，保存路径: {self.save_path}")

    def add_frame(self, frame):
        if self.is_recording and frame is not None:
            try:
                if len(frame.shape) == 1:
                    total_pixels = frame.shape[0]
                    if total_pixels % 3 == 0:
                        pixel_count = total_pixels // 3
                        possible_shapes = [
                            (self.resolution[1], self.resolution[0], 3),
                            (480, 640, 3),
                            (720, 1280, 3),
                            (1080, 1920, 3),
                            (240, 320, 3),
                            (360, 640, 3),
                        ]

                        for h, w, c in possible_shapes:
                            if h * w * c == total_pixels:
                                frame = frame.reshape(h, w, c)
                                break
                        else:
                            side_length = int(np.sqrt(pixel_count))
                            if side_length * side_length == pixel_count:
                                frame = frame.reshape(side_length, side_length, 3)
                            else:
                                self.logger.error(
                                    f"无法重塑1D数组, 像素总数: {total_pixels}, RGB像素数: {pixel_count}"
                                )
                                return
                    else:
                        self.logger.error(f"1D数组长度 {total_pixels} 不是3的倍数, 可能不是RGB数据")
                        return

                elif len(frame.shape) == 2:
                    frame = np.stack([frame, frame, frame], axis=2)

                elif len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    elif frame.shape[2] == 1:
                        frame = np.repeat(frame, 3, axis=2)
                    elif frame.shape[2] != 3:
                        self.logger.error(f"不支持的通道数: {frame.shape[2]}，跳过此帧")
                        return

                else:
                    self.logger.error(f"不支持的帧维度: {len(frame.shape)}，跳过此帧")
                    return

                if frame.dtype != np.uint8:
                    if frame.dtype == np.float32 or frame.dtype == np.float64:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)

                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    if self.writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        height, width = frame.shape[:2]
                        self.writer = cv2.VideoWriter(self.save_path, fourcc, self.fps, (width, height))
                        if not self.writer.isOpened():
                            self.logger.error(f"无法创建视频写入器: {self.save_path}")
                            self.writer = None
                            return
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.writer.write(frame_bgr)
                    self.frame_count += 1
                else:
                    self.logger.error(f"最终检查失败，帧格式不正确: {frame.shape}")

            except Exception as e:
                self.logger.exception(f"处理帧时出错: {e}")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False

        try:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            self.logger.info(f"视频已保存到: {self.save_path} (共{self.frame_count}帧)")

        except Exception as e:
            self.logger.error(f"保存视频时出错: {str(e)}")
        finally:
            self.frame_count = 0


class MultiDroneVideoRecorder:
    """多无人机视频录制器，为每个无人机单独录制视频"""

    def __init__(
        self,
        save_dir: str,
        num_drones: int,
        fps: int = 30,
        resolution: tuple = (640, 480),
        logger=None,
    ):
        self.save_dir = save_dir
        self.num_drones = num_drones
        self.fps = fps
        self.resolution = resolution
        self.logger = logger

        self.recorders = []
        for i in range(num_drones):
            save_path = os.path.join(save_dir, f"drone_{i}.mp4")
            recorder = VideoRecorder(save_path, fps, resolution, logger)
            self.recorders.append(recorder)

        self.is_recording = False
        os.makedirs(save_dir, exist_ok=True)

    def start_recording(self):
        self.is_recording = True
        for recorder in self.recorders:
            recorder.start_recording()
        self.logger.info(f"开始录制 {self.num_drones} 个无人机的视频")

    def add_frames(self, frames_list):
        if not self.is_recording:
            return

        if len(frames_list) != self.num_drones:
            self.logger.error(f"帧数量 {len(frames_list)} 与无人机数量 {self.num_drones} 不匹配")
            return

        for i, frame in enumerate(frames_list):
            if frame is not None:
                self.recorders[i].add_frame(frame)

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        for recorder in self.recorders:
            recorder.stop_recording()
        self.logger.info(f"已停止录制 {self.num_drones} 个无人机的视频")
