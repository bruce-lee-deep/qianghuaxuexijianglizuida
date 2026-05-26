#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import importlib
import os
import sys
import traceback
import gc

import numpy as np
import torch
from omni_drones import init_simulation_app
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl.envs import step_mdp
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import _terminated_or_truncated

from isaac_env.recording import CameraConfig, MultiCameraVideoWriter
from isaac_env.reward_state import RewardStateSnapshot
from tools.cluster_monitor import project_obstacle_num_label, project_obstacle_radius_label




def _flatten_done_flags(flags):
    if flags is None:
        return np.asarray([], dtype=bool)
    if isinstance(flags, np.ndarray):
        return flags.astype(bool).reshape(-1)
    flattened = np.asarray(flags, dtype=bool)
    return flattened.reshape(-1) if flattened.size > 0 else np.asarray([], dtype=bool)


def _project_obstacle_num_labels(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    projected = np.asarray([project_obstacle_num_label(item) for item in arr.reshape(-1)], dtype=np.float32)
    return projected.reshape(arr.shape)


def _project_obstacle_radius_labels(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    projected = np.asarray([project_obstacle_radius_label(item) for item in arr.reshape(-1)], dtype=np.float32)
    return projected.reshape(arr.shape)


class StandaloneDroneEnv:
    """独立的无人机环境包装器, 提供标准的step和reset接口。"""

    def __init__(
        self,
        cfg: DictConfig,
        enable_video_recording: bool = False,
        video_save_path: str = None,
        eval_mode: bool = False,
        logger=None,
    ):
        self.cfg = cfg
        self.logger = logger
        self.simulation_app = None
        self.env = None
        self.agent_spec = None
        self.current_state = None
        self.device = cfg.sim.device if hasattr(cfg.sim, "device") else "cuda"
        self.step_no = 0
        self.eval_mode = bool(eval_mode)

        self.camera_spec = None

        self.enable_video_recording = enable_video_recording
        self._record_fail_count = 0
        self._record_warmup_frames = 0
        self._record_frame_count = 0
        env_num = max(0, int(self.cfg.task.env.num_envs))
        self.recording_camera_count = min(env_num, CameraConfig.get_max_recording_camera_num())
        resolution_config = CameraConfig.get_resolution()
        self.recording_resolution = (resolution_config["width"], resolution_config["height"])
        self.multi_video_recorder = None
        self.reward_provider = None
        self._terminated_buffer = None
        self._truncated_buffer = None
        self._reward_buffer = None
        self._observation_buffer = None
        self._stats_buffer = {}
        if enable_video_recording and video_save_path:
            sim_dt = float(getattr(self.cfg.sim, "dt", 1.0 / 60.0))
            if sim_dt <= 0:
                sim_dt = 1.0 / 60.0
            control_freq_inv = float(OmegaConf.select(self.cfg, "task.control_freq_inv") or 1)
            fps = max(1.0, 1.0 / (sim_dt * control_freq_inv))
            self.multi_video_recorder = MultiDroneVideoRecorder(
                video_dir=video_save_path,
                num_drones=self.recording_camera_count,
                fps=fps,
                logger=self.logger,
            )

        self._init_env()

    @staticmethod
    def _resolve_algorithm_name() -> str:
        algo = os.environ.get("KAIWU_ALGORITHM", "ppo")
        algo = str(algo).strip().lower() or "ppo"
        if algo not in {"ppo", "diy"}:
            raise ValueError(f"Unsupported KAIWU_ALGORITHM: {algo}")
        return algo

    @staticmethod
    def _parse_reward_configs(usr_conf: dict | None) -> dict:
        rewards_section = usr_conf.get("rewards", {}) if isinstance(usr_conf, dict) else {}
        if not isinstance(rewards_section, dict):
            return {}

        configs = {}
        for term_name, term_cfg in rewards_section.items():
            if not isinstance(term_cfg, dict):
                continue
            configs[term_name] = {
                "weight": float(term_cfg.get("weight", 0.0)),
                "params": dict(term_cfg.get("params", {})),
            }
        return configs

    @classmethod
    def _resolve_reward_provider_module_name(cls, cfg: DictConfig) -> str:
        provider_name = OmegaConf.select(cfg, "task.reward_provider")
        if provider_name is not None:
            provider_name = str(provider_name).strip()
            if provider_name in {"agent_ppo", "agent_diy"}:
                return f"{provider_name}.feature.reward_process"
            raise ValueError(f"Unsupported reward_provider: {provider_name}")

        algo = cls._resolve_algorithm_name()
        return f"agent_{algo}.feature.reward_process"

    def _build_reward_provider(self):
        reward_configs = self._parse_reward_configs(OmegaConf.to_container(self.cfg.task.usr_conf, resolve=True))
        module_name = self._resolve_reward_provider_module_name(self.cfg)
        reward_module = importlib.import_module(module_name)
        reward_cls = getattr(reward_module, "RewardProcess")
        provider = reward_cls(
            reward_configs=reward_configs,
            task_config=OmegaConf.to_container(self.cfg.task, resolve=True),
            num_envs=int(self.cfg.task.env.num_envs),
            device=self.device,
        )
        return provider

    @staticmethod
    def _bind_reward_provider(provider, env):
        if provider is None or env is None:
            return
        initial_reward_state = RewardStateSnapshot.from_env(env)
        provider.bind_reward_state(initial_reward_state)
        # 将 provider 挂回运行中的任务环境，确保 _compute_reward_and_done()
        # 真正执行 PPO 自定义奖励，而不是落回空奖励分支。
        if hasattr(env, "reward_provider"):
            env.reward_provider = provider
        elif hasattr(env, "set_reward_provider"):
            env.set_reward_provider(provider)

    def _reset_eval_tracking(self):
        env_num = int(self.env.num_envs) if self.env is not None else 0
        self.eval_first_done_payloads = [None for _ in range(env_num)]

    @staticmethod
    def _done_mask_from_flags(terminated, truncated):
        terminated_flat = _flatten_done_flags(terminated)
        truncated_flat = _flatten_done_flags(truncated)
        if terminated_flat.size == 0:
            return truncated_flat
        if truncated_flat.size == 0:
            return terminated_flat
        if terminated_flat.size != truncated_flat.size:
            max_size = max(terminated_flat.size, truncated_flat.size)
            terminated_flat = np.resize(terminated_flat, max_size)
            truncated_flat = np.resize(truncated_flat, max_size)
        return np.logical_or(terminated_flat, truncated_flat)

    def _snapshot_eval_payloads(self, new_done_mask):
        task_env = self._get_task_env()
        if task_env is None:
            return
        latest_episode_payload = getattr(task_env, "latest_episode_payload", None)
        if not isinstance(latest_episode_payload, list):
            return

        for env_idx in np.flatnonzero(new_done_mask):
            if env_idx >= len(latest_episode_payload) or self.eval_first_done_payloads[env_idx] is not None:
                continue
            payload = latest_episode_payload[env_idx]
            if isinstance(payload, dict):
                self.eval_first_done_payloads[env_idx] = payload.copy()

    def _init_env(self):
        OmegaConf.register_new_resolver("eval", eval)
        OmegaConf.resolve(self.cfg)
        OmegaConf.set_struct(self.cfg, False)

        if self.enable_video_recording:
            os.environ["DISPLAY"] = ":0"
        else:
            os.environ.pop("DISPLAY", None)
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        os.environ["EGL_PLATFORM"] = "surfaceless"

        self.simulation_app = init_simulation_app(self.cfg)

        if self.enable_video_recording:
            import carb
            from omni.isaac.core.utils.extensions import enable_extension

            carb_settings_iface = carb.settings.get_settings()
            carb_settings_iface.set_bool("/orbit/offscreen_render/enabled", True)

            # Required extensions for Replicator annotator to produce frames
            enable_extension("omni.kit.viewport.rtx")
            enable_extension("omni.kit.viewport.pxr")
            enable_extension("omni.kit.viewport.bundle")
            enable_extension("omni.replicator.core")

        from omni_drones.envs.isaac_env import IsaacEnv

        env_class = IsaacEnv.REGISTRY[self.cfg.task.name]
        base_env = env_class(self.cfg, headless=self.cfg.headless)
        self.reward_provider = self._build_reward_provider()
        self._bind_reward_provider(self.reward_provider, base_env)

        if self.enable_video_recording:
            base_env.enable_render(True)
            base_env.enable_viewport = True

        transforms = [InitTracker()]
        action_transform = self.cfg.task.get("action_transform", None)
        if action_transform is not None:
            if action_transform == "rate":
                from omni_drones.controllers import RateController as _RateController
                from omni_drones.utils.torchrl.transforms import RateController

                controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
                transforms.append(RateController(controller))
            elif action_transform == "PIDrate":
                from omni_drones.controllers import PIDRateController as _PIDRateController
                from omni_drones.utils.torchrl.transforms import PIDRateController

                controller = _PIDRateController(self.cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
                transforms.append(PIDRateController(controller))

        self.env = TransformedEnv(base_env, Compose(*transforms)).train()
        self.env.set_seed(self.cfg.seed)
        self.agent_spec = self.env.agent_spec["drone"]

    def _get_task_env(self):
        if self.env is None:
            return None
        return getattr(self.env, "base_env", self.env)

    def clear_cached_episode_payloads(self):
        task_env = self._get_task_env()
        if task_env is None:
            return

        latest_episode_payload = getattr(task_env, "latest_episode_payload", None)
        if isinstance(latest_episode_payload, list):
            task_env.latest_episode_payload = [None for _ in latest_episode_payload]
        if self.eval_mode:
            env_num = len(latest_episode_payload) if isinstance(latest_episode_payload, list) else 0
            self.eval_first_done_payloads = [None for _ in range(env_num)]

    def reset(self, seed=None):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 仅首次 reset（step_no==0 表示环境刚创建）时设置种子，
        # 后续 reset 不再重置随机状态，让随机数序列自然递进以保证障碍物布局多样性。
        if self.step_no == 0 and seed is not None:
            self.env.set_seed(seed)
        self.current_state = self.env.reset()
        self.step_no = 0
        terminated = self._extract_terminated(self.current_state)
        truncated = self._extract_truncated(self.current_state)
        reset_obs = self._extract_observation(self.current_state)

        if self.eval_mode:
            self._reset_eval_tracking()

        if self.multi_video_recorder:
            self.multi_video_recorder.stop_recording()
            self._record_frame_count = 0
            self.multi_video_recorder.start_recording()

        return self.current_state.detach().cpu(), terminated, truncated

    def step(self, action=None):
        self.step_no += 1

        action_state = self.format_action(action)

        with torch.no_grad():
            stepped_data = self.env.step(action_state)

            # 在 reset 之前提取 stats/reward/terminated/truncated，
            # 因为 env.reset() 会就地清零 done env 的 stats（TensorDict 引用语义）
            next_data = stepped_data.get("next", stepped_data)
            reward = self._extract_reward(next_data)
            truncated = self._extract_truncated(next_data)
            terminated = self._extract_terminated(next_data)
            stats = self._extract_stats(next_data)
            if self.eval_mode:
                done_mask = self._done_mask_from_flags(terminated, truncated)
                first_done_payloads = getattr(self, "eval_first_done_payloads", None)
                if not isinstance(first_done_payloads, list) or len(first_done_payloads) != done_mask.size:
                    self.eval_first_done_payloads = [None for _ in range(done_mask.size)]
                    first_done_payloads = self.eval_first_done_payloads
                seen_done_mask = np.asarray([payload is not None for payload in first_done_payloads], dtype=bool)
                new_done_mask = done_mask & ~seen_done_mask
                if new_done_mask.any():
                    self._snapshot_eval_payloads(new_done_mask)
            # 提取完毕后立即断开 next_data 对 stepped_data 内部 tensor 的引用
            del next_data

            next_stepped_data = step_mdp(stepped_data)
            # stepped_data 的数据已被 step_mdp 搬运，立即释放避免双份内存
            del stepped_data
            any_done = _terminated_or_truncated(next_stepped_data)
            if any_done:
                try:
                    next_stepped_data = self.env.reset(next_stepped_data)
                    if next_stepped_data is None:
                        print("[ERROR] env.reset() returned None! Keeping previous state.")
                        # fallback: 无法恢复 stepped_data，保留上一帧状态
                        next_stepped_data = self.current_state
                except Exception as e:
                    print(f"[ERROR] env.reset() raised exception: {e}")
                    traceback.print_exc()
                    raise
        if self.multi_video_recorder and self.multi_video_recorder.is_recording and self.camera_spec is not None:
            self._record_frame()

        obs = self._extract_observation(next_stepped_data)
        # 替换 current_state 前显式释放旧引用
        old_state = self.current_state
        self.current_state = next_stepped_data
        del old_state

        # 每 500 步清理一次 CUDA 缓存碎片（不影响 GPU tensor 正确性）
        if self.step_no % 500 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.step_no, obs, reward, terminated, truncated, stats

    def _frame_dict_to_numpy(self, frame_dict):
        import numpy as np

        if "rgb" in frame_dict.keys():
            rgb_tensor = frame_dict["rgb"]
        elif "rgba" in frame_dict.keys():
            rgb_tensor = frame_dict["rgba"][:3, :, :]
        else:
            available_keys = list(frame_dict.keys())
            if not available_keys:
                raise ValueError("TensorDict中没有可用的图像数据")
            rgb_tensor = frame_dict[available_keys[0]]

        if not isinstance(rgb_tensor, torch.Tensor):
            raise TypeError("图像数据不是torch.Tensor类型")

        frame_np = rgb_tensor.permute(1, 2, 0).detach().cpu().numpy()
        del rgb_tensor

        if frame_np.dtype == np.float32 or frame_np.dtype == np.float64:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
        elif frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)

        return frame_np

    def _resolve_recording_camera_pose(self):
        viewer_eye = OmegaConf.select(self.cfg, "viewer.eye")
        viewer_lookat = OmegaConf.select(self.cfg, "viewer.lookat")
        if viewer_eye is not None and viewer_lookat is not None:
            return {
                "position": tuple(float(v) for v in viewer_eye),
                "target": tuple(float(v) for v in viewer_lookat),
                "description": "viewer 局部录屏视角",
            }

        camera_preset = OmegaConf.select(self.cfg, "video_recording.camera_preset") or CameraConfig.get_default_preset_name()
        arena_cfg = OmegaConf.select(self.cfg, "task.arena")
        return CameraConfig.get_preset(camera_preset, arena_size=arena_cfg)

    def _record_frame(self):
        import numpy as np

        try:
            frames_tensordict = self.camera_spec.get_images()

            if frames_tensordict is None or len(frames_tensordict) == 0:
                self._log_record_error("get_images返回了None或空数据")
                return

            if not self.multi_video_recorder or not self.multi_video_recorder.recorders:
                self._log_record_error("录制器不可用")
                return

            num_cams_per_env = getattr(self, "_camera_count_per_env", 1)
            total_cams = len(frames_tensordict)
            env_count = total_cams // num_cams_per_env

            if env_count == 0:
                self._log_record_error("没有可用的录屏相机")
                return

            layout = CameraConfig.get_camera_layout()
            rows, cols = layout["rows"], layout["cols"]
            positions = layout["positions"]

            frames_list = []
            for env_idx in range(min(env_count, len(self.multi_video_recorder.recorders))):
                cam_frames = []
                for cam_idx in range(num_cams_per_env):
                    fidx = env_idx * num_cams_per_env + cam_idx
                    cam_frames.append(self._frame_dict_to_numpy(frames_tensordict[fidx]))

                if num_cams_per_env == 1:
                    frames_list.append(cam_frames[0])
                else:
                    # Arrange cameras into a grid (left-to-right, top-to-bottom)
                    h = min(f.shape[0] for f in cam_frames)
                    w = min(f.shape[1] for f in cam_frames)
                    cam_frames = [f[:h, :w] for f in cam_frames]

                    # Build each row by horizontally concatenating, then vertically stack rows
                    grid = np.zeros((h * rows, w * cols, cam_frames[0].shape[2]), dtype=cam_frames[0].dtype)
                    for cam_idx, (row, col) in enumerate(positions):
                        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = cam_frames[cam_idx]
                    frames_list.append(grid)

            while len(frames_list) < len(self.multi_video_recorder.recorders):
                frames_list.append(None)

            self.multi_video_recorder.add_frames(frames_list)
            self._record_fail_count = 0
            del frames_tensordict

        except Exception as e:
            self._log_record_error(f"录制帧时出错: {e}", exc_info=True)

    def _log_record_error(self, msg: str, exc_info: bool = False):
        if not hasattr(self, "_record_fail_count"):
            self._record_fail_count = 0
        self._record_fail_count += 1

        if self._record_fail_count > 5 and self._record_fail_count % 50 != 0:
            return

        suffix = ""
        if self._record_fail_count > 5:
            suffix = f" (连续失败 {self._record_fail_count} 次，后续每50次报告一次)"

        full_msg = f"[视频录制] {msg}{suffix}"
        if exc_info:
            self.logger.exception(full_msg)
        else:
            self.logger.error(full_msg)

        print(full_msg, file=sys.stderr, flush=True)
        if exc_info:
            traceback.print_exc(file=sys.stderr)

    def create_camera(self):
        from omni_drones.sensors.camera import Camera
        from omni_drones.sensors.config import PinholeCameraCfg

        arena_cfg = OmegaConf.select(self.cfg, "task.arena")
        camera_presets = CameraConfig.get_camera_presets(arena_size=arena_cfg)
        resolution_config = CameraConfig.get_resolution()
        usd_params = CameraConfig.get_usd_params()

        if self.enable_video_recording:
            try:
                if hasattr(self.env.base_env, "enable_render"):
                    self.env.base_env.enable_render(True)
                    self.logger.info("已启用环境渲染用于视频录制")
            except Exception as e:
                self.logger.error(f"启用渲染时出错: {e}")

            self.logger.info(f"环境初始化完成: 环境数量: {self.env.num_envs}, 设备: {self.device}")

        camera_count = min(self.recording_camera_count, self.env.num_envs)
        self.recording_camera_count = camera_count
        if camera_count <= 0:
            self.logger.warning("未创建录屏相机：当前没有可录制的env")
            return
        if self.env.num_envs > camera_count:
            self.logger.info(f"当前共有 {self.env.num_envs} 个env，仅录制前 {camera_count} 个env")

        # Create cameras per env (1~4, arranged in grid)
        self._camera_count_per_env = len(camera_presets)
        prim_paths = []
        translations = []
        targets = []
        for env_idx in range(camera_count):
            for cam_idx, preset in enumerate(camera_presets):
                cam_pos = tuple(float(v) for v in preset["position"])
                cam_tgt = tuple(float(v) for v in preset["target"])
                prim_path = f"/World/envs/env_{env_idx}/Camera_record_{env_idx}_{cam_idx}"
                prim_paths.append(prim_path)
                translations.append(cam_pos)
                targets.append(cam_tgt)

        cam_w = resolution_config["width"]
        cam_h = resolution_config["height"]
        cfg = PinholeCameraCfg(
            sensor_tick=0,
            resolution=(cam_w, cam_h),
            data_types=["rgb"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=usd_params["focal_length"],
                focus_distance=usd_params["focus_distance"],
                horizontal_aperture=usd_params["horizontal_aperture"],
                clipping_range=usd_params["clipping_range"],
            ),
        )
        self.camera_spec = Camera(cfg=cfg)
        self.camera_spec.spawn(prim_paths, translations=translations, targets=targets)
        self.camera_spec.initialize("/World/envs/.*/Camera_record_.*")

        warmup_frames = CameraConfig.get_warmup_frames(camera_count=camera_count * self._camera_count_per_env)
        layout = CameraConfig.get_camera_layout()
        output_res = CameraConfig.get_output_resolution()
        self.logger.info(
            f"多摄像机预渲染 {warmup_frames} 帧以收敛光照，单路分辨率: {cam_w}x{cam_h}，"
            f"拼接后: {output_res['width']}x{output_res['height']}"
        )
        sim = self.camera_spec.sim
        for _ in range(warmup_frames):
            sim.render()

        # Apply VRAM-saving rendering optimizations AFTER warmup,
        # so the RTX pipeline is fully initialized before we modify settings.
        CameraConfig.apply_render_optimizations()

        preset_descs = [p["description"] for p in camera_presets]
        self.logger.info(
            f"多摄像机初始化完成（{camera_count} 个env × {self._camera_count_per_env} 路，"
            f"布局: {layout['rows']}×{layout['cols']}），视角: {preset_descs}"
        )

    def _extract_terminated(self, tensordict):
        if ("terminated") in tensordict.keys(include_nested=True):
            return self._extract_to_cpu_buffer(tensordict["terminated"], "_terminated_buffer")
        return None

    def _extract_truncated(self, tensordict):
        if ("truncated") in tensordict.keys(include_nested=True):
            return self._extract_to_cpu_buffer(tensordict["truncated"], "_truncated_buffer")
        return None

    def _extract_reward(self, tensordict):
        if ("agents", "reward") in tensordict.keys(include_nested=True):
            return self._extract_to_cpu_buffer(tensordict["agents", "reward"], "_reward_buffer")
        return None

    def _extract_observation(self, tensordict):
        if ("agents", "observation") in tensordict.keys(include_nested=True):
            return self._extract_to_cpu_buffer(tensordict["agents", "observation"], "_observation_buffer")
        return None

    def _extract_to_cpu_buffer(self, value, buffer_attr: str):
        """Copy tensor data into a reusable CPU torch buffer.

        Avoids allocating a fresh CPU tensor / numpy array on every env step
        for hot-path outputs such as observation/reward/done flags.
        """
        if not isinstance(value, torch.Tensor):
            return value

        src = value.detach()
        cpu_tensor = getattr(self, buffer_attr, None)
        if (
            cpu_tensor is None
            or tuple(cpu_tensor.shape) != tuple(src.shape)
            or cpu_tensor.dtype != src.dtype
        ):
            cpu_tensor = torch.empty(src.shape, dtype=src.dtype, device="cpu")
            setattr(self, buffer_attr, cpu_tensor)

        cpu_tensor.copy_(src, non_blocking=False)
        return cpu_tensor.numpy()

    @staticmethod
    def _copy_tensor_into_cpu_tensor(value, cpu_tensor: torch.Tensor | None):
        """Return a reused CPU tensor containing the latest tensor value."""
        if not isinstance(value, torch.Tensor):
            return value

        src = value.detach()
        if (
            cpu_tensor is None
            or tuple(cpu_tensor.shape) != tuple(src.shape)
            or cpu_tensor.dtype != src.dtype
        ):
            cpu_tensor = torch.empty(src.shape, dtype=src.dtype, device="cpu")

        cpu_tensor.copy_(src, non_blocking=False)
        return cpu_tensor

    def _extract_stats(self, tensordict):
        """提取 stats 到预分配的 CPU torch buffer 池中。"""
        if "stats" not in tensordict.keys(include_nested=True):
            return None
        try:
            stats_td = tensordict["stats"]

            for key in stats_td.keys():
                val = stats_td[key]
                if isinstance(val, torch.Tensor):
                    cpu_tensor = self._copy_tensor_into_cpu_tensor(val, self._stats_buffer.get(key))
                    self._stats_buffer[key] = cpu_tensor
                    arr = cpu_tensor.numpy()
                    if key == "obstacle_num":
                        arr = _project_obstacle_num_labels(arr)
                    elif key == "obstacle_radius":
                        arr = _project_obstacle_radius_labels(arr)
                    if key in {"obstacle_num", "obstacle_radius"}:
                        cpu_tensor.copy_(torch.from_numpy(arr), non_blocking=False)
                else:
                    self._stats_buffer[key] = val

            # 返回浅拷贝 dict（key→buffer 引用，不新建 CPU tensor）。
            return {
                key: value.numpy() if isinstance(value, torch.Tensor) else value
                for key, value in self._stats_buffer.items()
            }
        except Exception as exc:
            raise RuntimeError("提取 stats 失败") from exc

    def get_observation_space(self):
        return self.agent_spec.observation_spec

    def get_action_space(self):
        return self.agent_spec.action_spec

    def get_num_envs(self):
        return self.env.num_envs

    def stop_recording(self):
        if self.multi_video_recorder:
            self.multi_video_recorder.stop_recording()

    def finalize_recording(self):
        if self.multi_video_recorder:
            self.multi_video_recorder.stop_recording()

    def close(self):
        self.stop_recording()

        env = self.env
        self.current_state = None
        self.agent_spec = None
        self._terminated_buffer = None
        self._truncated_buffer = None
        self._reward_buffer = None
        self._observation_buffer = None
        self._stats_buffer = {}

        # 先释放 recorder/camera 对 USD / replicator / viewport 的引用，
        # 再关闭 env 和 SimulationApp，避免 Isaac Sim 在 shutdown 阶段发生析构顺序问题。
        self.multi_video_recorder = None
        self.camera_spec = None

        if env is not None:
            try:
                if hasattr(env, "close"):
                    env.close()
            except Exception as exc:
                raise RuntimeError("关闭底层环境失败") from exc
            finally:
                self.env = None
        else:
            self.env = None

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                raise RuntimeError("CUDA synchronize 失败") from exc
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                raise RuntimeError("CUDA empty_cache 失败") from exc

        if self.simulation_app is not None:
            simulation_app = self.simulation_app
            self.simulation_app = None
            simulation_app.close()

    def rand_action(self):
        return self.env.rand_action(self.current_state)

    def format_action(self, action):
        if self.current_state is None:
            print("[ERROR] current_state is None! Stack trace:")
            traceback.print_stack()
            raise RuntimeError("current_state is None - reset() was not called or failed")
        action_dict = self.tensor_to_action_tensordict(action)
        self.current_state.update(action_dict)
        return self.current_state

    def tensor_to_action_tensordict(self, action_tensor: torch.Tensor) -> TensorDict:
        if action_tensor.device != torch.device(self.device):
            action_tensor = action_tensor.to(self.device)
        if action_tensor.dtype != torch.float32:
            action_tensor = action_tensor.to(torch.float32)

        batch_size = action_tensor.shape[0]
        return TensorDict(
            {
                "agents": TensorDict(
                    {"action": action_tensor},
                    batch_size=torch.Size([batch_size]),
                    device=self.device,
                )
            },
            batch_size=torch.Size([batch_size]),
            device=self.device,
        )


class MultiDroneVideoRecorder(MultiCameraVideoWriter):
    """按无人机/env 命名的视频写入器。"""

    def __init__(self, video_dir: str, num_drones: int, fps: float = 30.0, logger=None, resolution=None):
        super().__init__(video_dir=video_dir, num_cameras=num_drones, fps=fps, logger=logger)
        self.num_drones = num_drones
        self.resolution = resolution
