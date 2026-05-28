#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Convert desired velocity into 4D drone action [roll_rate, pitch_rate, yaw_rate, thrust].
环境使用 PIDrate (CTBR) 控制器，动作范围 [-1, 1]。
"""

from __future__ import annotations

import torch

from agent_ppo.conf.conf import Config


class VelocityController:
    """Map desired velocity to PIDrate action commands.

    action[0] = roll_rate   [-1,1] → [-180, +180]°/s
    action[1] = pitch_rate  [-1,1] → [-180, +180]°/s
    action[2] = yaw_rate    [-1,1] → [-180, +180]°/s
    action[3] = thrust      [-1,1] → (x+1)/2 → [0,1]
    """

    def __init__(self):
        self.kp_xy = Config.VC_KP_XY
        self.kp_z = Config.VC_KP_Z

    def compute_action(
        self,
        linear_velocity: torch.Tensor,     # [batch, 3]  body-frame
        desired_velocity: torch.Tensor,    # [batch, 3]  world-frame desired
        rotation_matrix: torch.Tensor,     # [batch, 3, 3]
        takeoff: bool = False,
    ) -> torch.Tensor:
        """Return 4D action tensor [batch, 4] in [-1, 1]."""
        batch_size = linear_velocity.shape[0]
        device = linear_velocity.device

        # Current velocity in world frame: v_world = R @ v_body
        v_body = linear_velocity
        v_world = torch.bmm(rotation_matrix, v_body.unsqueeze(-1)).squeeze(-1)

        # Velocity error (world frame).
        v_err = desired_velocity - v_world  # [batch, 3]

        # Convert world-frame velocity error to body-frame rate commands.
        # To move forward (+x body): need negative pitch rate (nose down).
        # To move right (+y body): need positive roll rate (right roll).
        # To climb (+z world): need more thrust.
        R_T = rotation_matrix.transpose(1, 2)  # world → body
        v_err_body = torch.bmm(R_T, v_err.unsqueeze(-1)).squeeze(-1)

        # Body-frame desired rates.
        # v_err_body[0] = forward error → nose down = negative pitch → -kp * v_err_body[0]
        # v_err_body[1] = right error   → right roll = positive roll → +kp * v_err_body[1]
        roll_rate = self.kp_xy * v_err_body[:, 1]
        pitch_rate = -self.kp_xy * v_err_body[:, 0]
        yaw_rate = torch.zeros(batch_size, device=device)

        # Thrust: gravity offset + vertical control.
        thrust = Config.VC_HOVER_OFFSET + self.kp_z * v_err[:, 2]
        if takeoff:
            thrust = thrust + 0.2  # extra for takeoff

        # Clamp to [-1, 1].
        roll_rate = torch.clamp(roll_rate, -1.0, 1.0)
        pitch_rate = torch.clamp(pitch_rate, -1.0, 1.0)
        yaw_rate = torch.clamp(yaw_rate, -1.0, 1.0)
        thrust = torch.clamp(thrust, -1.0, 1.0)

        action = torch.stack([roll_rate, pitch_rate, yaw_rate, thrust], dim=-1)
        return action

    def compute_hover_action(
        self,
        linear_velocity: torch.Tensor,
        rotation_matrix: torch.Tensor,
        position_error: torch.Tensor,   # [batch, 3] world-frame error to target
    ) -> torch.Tensor:
        """Hover position hold: stronger damping, weaker position pull."""
        # Convert position error to desired velocity (very conservative).
        desired_vel = 0.5 * position_error
        desired_vel = torch.clamp(desired_vel, -0.3, 0.3)
        return self.compute_action(linear_velocity, desired_vel, rotation_matrix)


__all__ = ["VelocityController"]
