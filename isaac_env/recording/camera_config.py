#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
摄像机配置文件。
"""


class CameraConfig:
    """摄像机配置类，提供录屏视角、镜头参数与环境间距建议。"""

    DEFAULT_PRESET = "oblique_30"

    PRESETS = {
        "default": {
            "position": (2.5, -5.5, 8.5),
            "target": (2.5, 2.5, 0.5),
            "description": "约45度斜俯视，摄像机对准场地正中心",
        },
        "oblique_45": {
            "position": (2.5, -5.5, 8.5),
            "target": (2.5, 2.5, 0.5),
            "description": "约45度斜俯视，摄像机对准场地正中心",
        },
        "oblique_30": {
            "position": (2.5, -8.0, 5.5),
            "target": (2.5, 2.5, 0.5),
            "description": "约30度低角度斜视，摄像机更贴近地面",
        },
        "oblique_overview": {
            "position": (2.5, -3.0, 9.0),
            "target": (2.5, 2.5, 0.5),
            "description": "高角度俯视全景，摄像机对准场地正中心",
        },
        "close_up": {
            "position": (2.5, -6.0, 4.5),
            "target": (2.5, 2.5, 0.5),
            "description": "约30度低角度近距斜视，比oblique_30更靠近场景中心",
        },
        "close_up_behind": {
            "position": (2.5, 11.0, 4.5),
            "target": (2.5, 2.5, 0.5),
            "description": "约30度低角度近距斜视（背侧），从场景内侧往外看",
        },
        "top_down": {
            "position": (2.5, 2.49, 10.0),
            "target": (2.5, 2.5, 0.0),
            "description": "正上方完全俯视，适合观察路径规划",
        },
    }

    RESOLUTION = {"width": 1080, "height": 1080}
    USD_PARAMS = {
        "focal_length": 24.0,
        "focus_distance": 400.0,
        "horizontal_aperture": 20.955,
        "clipping_range": (0.1, 1.0e5),
    }

    # --- Rendering optimizations to reduce VRAM during recording ---
    # Applied AFTER camera warmup (pipeline fully initialized) so they
    # only affect rendering quality, not pipeline construction.
    RENDER_OPTIMIZATIONS = {
        # Disable expensive lighting/reflection features
        "/rtx/reflections/enabled": False,
        "/rtx/indirectDiffuse/enabled": False,
        "/rtx/ambientOcclusion/enabled": False,
        "/rtx/shadows/enabled": False,
        "/rtx/translucency/enabled": False,
        "/rtx/subsurface/enabled": False,
        "/rtx/denoiser/enabled": False,
        # Anti-aliasing: 0=off
        "/rtx/post/aa/op": 0,
        # Reduce sample counts for direct lighting
        "/rtx/directLighting/sampledLighting/enabled": True,
        "/rtx/directLighting/sampledLighting/maxSpp": 1,
    }

    WARMUP_FRAMES = 20
    MAX_CAMERA_NUM = 8
    MAX_RECORDING_CAMERA_NUM = 1

    # Recording camera presets: 1~4 cameras, arranged left-to-right then top-to-bottom.
    # Examples:
    #   CAMERA_PRESETS = ["close_up_behind"]                      → 1 camera, full frame
    #   CAMERA_PRESETS = ["close_up_behind", "close_up"]          → 2 cameras, side by side
    #   CAMERA_PRESETS = ["top_down", "oblique_45", "oblique_30"] → 3 cameras (2 top + 1 bottom centered)
    #   CAMERA_PRESETS = ["A", "B", "C", "D"]                     → 4 cameras (2×2 grid)
    CAMERA_PRESETS = ["top_down", "oblique_30"]

    @classmethod
    def get_default_preset_name(cls):
        return cls.DEFAULT_PRESET

    @classmethod
    def get_max_camera_num(cls):
        return cls.MAX_CAMERA_NUM

    @classmethod
    def get_max_recording_camera_num(cls):
        return cls.MAX_RECORDING_CAMERA_NUM

    @classmethod
    def get_resolution(cls):
        return cls.RESOLUTION.copy()

    @classmethod
    def get_usd_params(cls):
        return cls.USD_PARAMS.copy()

    @classmethod
    def apply_render_optimizations(cls):
        """Apply carb settings to reduce VRAM usage during video recording.

        Disables reflections, shadows, AO, denoiser, post-processing, etc.
        Call this after carb is available and before camera creation.
        """
        try:
            import carb
            settings = carb.settings.get_settings()
            for path, value in cls.RENDER_OPTIMIZATIONS.items():
                if isinstance(value, bool):
                    settings.set_bool(path, value)
                elif isinstance(value, int):
                    settings.set_int(path, value)
                elif isinstance(value, float):
                    settings.set_float(path, value)
                elif isinstance(value, str):
                    settings.set_string(path, value)
        except Exception as exc:
            raise RuntimeError("应用渲染优化配置失败") from exc

    @classmethod
    def get_warmup_frames(cls, camera_count: int = 1):
        return cls.WARMUP_FRAMES

    @classmethod
    def get_camera_presets(cls, arena_size=None):
        """Return a list of camera pose configs for recording."""
        return [cls.get_preset(name, arena_size=arena_size) for name in cls.CAMERA_PRESETS]

    @classmethod
    def get_camera_layout(cls):
        """Calculate the grid layout for CAMERA_PRESETS.

        Returns a dict with:
            rows, cols: grid dimensions
            positions: list of (row, col) for each camera (row 0 = top)
        Layout rules:
            1 cam  → 1×1
            2 cams → 1×2 (side by side)
            3 cams → 2×2 (2 on top row, 1 centered on bottom)
            4 cams → 2×2
        """
        n = len(cls.CAMERA_PRESETS)
        if n <= 0:
            return {"rows": 0, "cols": 0, "positions": []}
        if n == 1:
            return {"rows": 1, "cols": 1, "positions": [(0, 0)]}
        if n == 2:
            return {"rows": 1, "cols": 2, "positions": [(0, 0), (0, 1)]}
        # 3 or 4 cameras: 2×2 grid
        if n == 3:
            return {"rows": 2, "cols": 2, "positions": [(0, 0), (0, 1), (1, 0)]}
        # n == 4
        return {"rows": 2, "cols": 2, "positions": [(0, 0), (0, 1), (1, 0), (1, 1)]}

    @classmethod
    def get_output_resolution(cls):
        """Calculate the final output resolution based on camera layout and single-cam resolution."""
        layout = cls.get_camera_layout()
        rows, cols = layout["rows"], layout["cols"]
        w, h = cls.RESOLUTION["width"], cls.RESOLUTION["height"]
        return {"width": w * cols, "height": h * rows}

    @classmethod
    def _normalize_arena_size(cls, arena_size):
        if arena_size is None:
            return None

        try:
            if hasattr(arena_size, "get") and "L" in arena_size:
                length = float(arena_size.get("L", 0.0))
                width = float(arena_size.get("W", 0.0))
                height = float(arena_size.get("H", 0.0))
            else:
                length = float(arena_size[0])
                width = float(arena_size[1])
                height = float(arena_size[2]) if len(arena_size) > 2 else 3.0
        except (TypeError, ValueError, IndexError):
            return None

        if length <= 0 or width <= 0:
            return None

        return {
            "L": length,
            "W": width,
            "H": max(height, 1.0),
        }

    @classmethod
    def _build_adaptive_preset(cls, preset_name: str, arena_size):
        dims = cls._normalize_arena_size(arena_size)
        if dims is None:
            return None

        length = dims["L"]
        width = dims["W"]
        height = dims["H"]
        max_side = max(length, width)
        center_x = round(length / 2.0, 2)
        center_y = round(width / 2.0, 2)
        target_z = round(min(max(height * 0.25, 0.7), height), 2)

        adaptive_presets = {
            "default": {
                "position": (
                    center_x,
                    round(center_y - max_side * 0.8, 2),
                    round(height + max_side * 0.8, 2),
                ),
                "target": (center_x, center_y, target_z),
                "description": f"约45度斜俯视（按 {length:.1f}x{width:.1f}x{height:.1f} 场地自适应）",
            },
            "oblique_45": {
                "position": (
                    center_x,
                    round(center_y - max_side * 0.8, 2),
                    round(height + max_side * 0.8, 2),
                ),
                "target": (center_x, center_y, target_z),
                "description": f"约45度斜俯视（按 {length:.1f}x{width:.1f}x{height:.1f} 场地自适应）",
            },
            "oblique_30": {
                "position": (
                    center_x,
                    round(center_y - max_side * 1.2, 2),
                    round(height + max_side * 0.35, 2),
                ),
                "target": (center_x, center_y, target_z),
                "description": f"约30度低角度斜视（按 {length:.1f}x{width:.1f}x{height:.1f} 场地自适应）",
            },
            "close_up": {
                "position": (
                    center_x,
                    round(center_y - max_side * 0.8, 2),
                    round(height + max_side * 0.25, 2),
                ),
                "target": (center_x, center_y, target_z),
                "description": f"约30度低角度近距斜视（按 {length:.1f}x{width:.1f}x{height:.1f} 场地自适应）",
            },
            "close_up_behind": {
                "position": (
                    center_x,
                    round(center_y + max_side * 0.8, 2),
                    round(height + max_side * 0.25, 2),
                ),
                "target": (center_x, center_y, target_z),
                "description": f"约30度低角度近距斜视-背侧（按 {length:.1f}x{width:.1f}x{height:.1f} 场地自适应）",
            },
            "top_down": {
                "position": (center_x, round(center_y - 0.01, 4), round(height + max_side * 1.4, 2)),
                "target": (center_x, center_y, 0.0),
                "description": f"正上方完全俯视（按 {length:.1f}x{width:.1f}x{height:.1f} 场地自适应）",
            },
        }
        return adaptive_presets.get(preset_name)

    @classmethod
    def get_preset(cls, preset_name: str, arena_size=None):
        if preset_name not in cls.PRESETS:
            print(f"未找到预设 '{preset_name}'，可用预设: {list(cls.PRESETS.keys())}")
            preset_name = cls.DEFAULT_PRESET

        adaptive_config = cls._build_adaptive_preset(preset_name, arena_size)
        if adaptive_config is not None:
            return adaptive_config

        return cls.PRESETS[preset_name].copy()

    @classmethod
    def get_recommended_env_spacing(cls, arena_size, padding: float = 4.0, minimum: float = 12.0):
        if arena_size is None:
            return minimum

        if isinstance(arena_size, dict):
            length = float(arena_size.get("L", minimum))
            width = float(arena_size.get("W", minimum))
        else:
            length = float(arena_size[0])
            width = float(arena_size[1])

        max_side = max(length, width)
        return max(minimum, round(max_side + padding, 2))

    @classmethod
    def list_presets(cls):
        print("可用的摄像机预设配置:")
        for name, config in cls.PRESETS.items():
            pos = config["position"]
            target = config["target"]
            desc = config["description"]
            print(f"  {name:18} - 位置: {pos}, 目标: {target}")
            print(f"  {' ':18}   {desc}")
            print()

    @classmethod
    def create_custom_config(cls, position: tuple, target: tuple, description: str = "自定义配置"):
        return {"position": position, "target": target, "description": description}


if __name__ == "__main__":
    CameraConfig.list_presets()
    config = CameraConfig.get_preset(CameraConfig.get_default_preset_name())
    print(f"默认录屏配置: {config}")
    custom = CameraConfig.create_custom_config(position=(12, -8, 6), target=(5, 3, 1), description="自定义观察角度")
    print(f"自定义配置: {custom}")
