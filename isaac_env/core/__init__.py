#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

__all__ = ["Drone", "StandaloneDroneEnv"]


def __getattr__(name):
    if name == "Drone":
        from .arena_env import Drone

        return Drone
    if name == "StandaloneDroneEnv":
        from .drone_runtime import StandaloneDroneEnv

        return StandaloneDroneEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
