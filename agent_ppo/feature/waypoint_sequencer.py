#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Waypoint sequencer using nearest-neighbor heuristic + 2-opt local optimization.
使用贪心最近邻 + 2-opt 局部优化的途径点排序器。
"""

from __future__ import annotations

import torch

from agent_ppo.conf.conf import Config


class WaypointSequencer:
    """Plan the optimal order to visit waypoints, ending at the goal.

    Uses nearest-neighbor greedy construction followed by 2-opt refinement
    to produce a near-optimal TSP tour over the waypoints + goal.

    Usage:
        seq = WaypointSequencer()
        targets = seq.plan(drone_pos, waypoint_pos, visited, active, goal_pos)
        # targets: [batch, max_targets, 3], ordered waypoints followed by goal
    """

    def __init__(self):
        self._max_wp = Config.MAX_WAYPOINTS
        # Internal state: which waypoint the drone is currently heading toward
        # shape: [batch]
        self._current_target_idx = None

    def plan(
        self,
        drone_pos: torch.Tensor,          # [batch, 3]
        waypoint_pos: torch.Tensor,        # [batch, max_wp, 3]
        waypoint_visited: torch.Tensor,    # [batch, max_wp] bool
        waypoint_active: torch.Tensor,     # [batch, max_wp] bool
        goal_pos: torch.Tensor,            # [batch, 3]
    ):
        """Return ordered target positions [batch, max_targets, 3].

        The ordered list contains unvisited active waypoints in the planned
        visit order, followed by the goal position as the final target.
        """
        batch_size = drone_pos.shape[0]
        device = drone_pos.device
        max_wp = waypoint_pos.shape[1]

        # Determine which waypoints are pending (active & not visited).
        pending_mask = waypoint_active & (~waypoint_visited)  # [batch, max_wp]

        # Build per-env ordered list.
        all_ordered = []
        max_len = 0

        for b in range(batch_size):
            pending_idx = pending_mask[b].nonzero(as_tuple=False).view(-1).tolist()
            if not pending_idx:
                # No waypoints to visit, target is just the goal.
                all_ordered.append([goal_pos[b].tolist()])
                max_len = max(max_len, 1)
                continue

            # Start from nearest waypoint to drone.
            current_pos = drone_pos[b]
            remaining = list(pending_idx)
            ordered = []

            while remaining:
                # Nearest-neighbor selection.
                positions = waypoint_pos[b, remaining]  # [n, 3]
                dists = torch.norm(positions - current_pos, dim=-1)
                nearest_local = dists.argmin()
                chosen = remaining.pop(nearest_local.item())
                ordered.append(waypoint_pos[b, chosen].tolist())
                current_pos = waypoint_pos[b, chosen]

            # 2-opt local optimization.
            ordered = self._two_opt(ordered)

            # Append goal as final target.
            ordered.append(goal_pos[b].tolist())
            all_ordered.append(ordered)
            max_len = max(max_len, len(ordered))

        # Pad to uniform length.
        padded = torch.zeros(batch_size, max_len, 3, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for b, targets in enumerate(all_ordered):
            n = len(targets)
            padded[b, :n] = torch.tensor(targets, device=device)
            lengths[b] = n

        return padded, lengths  # [batch, max_targets, 3], [batch]

    @staticmethod
    def _two_opt(route):
        """2-opt local search for TSP tour improvement.

        Args:
            route: List of [x, y, z] positions.

        Returns:
            Improved route as list of positions.
        """
        if len(route) <= 2:
            return route

        n = len(route)

        def _dist(a, b):
            return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # Check if reversing segment [i+1, j] improves total length.
                    old_len = _dist(route[i], route[i + 1]) + _dist(route[j], route[(j + 1) % n])
                    new_len = _dist(route[i], route[j]) + _dist(route[i + 1], route[(j + 1) % n])
                    if new_len < old_len - 1e-6:
                        route[i + 1 : j + 1] = reversed(route[i + 1 : j + 1])
                        improved = True
                        break
                if improved:
                    break

        return route

    def get_current_target(
        self,
        drone_pos: torch.Tensor,           # [batch, 3]
        ordered_targets: torch.Tensor,      # [batch, max_targets, 3]
        current_idx: torch.Tensor = None,   # [batch] int
        waypoint_collect_radius: float = 0.3,
    ):
        """Advance the target pointer if the current one has been reached.

        Returns:
            current_target: [batch, 3] — current target position.
            current_idx: [batch] — updated target index.
            done: [batch] bool — all targets reached (episode complete).
        """
        batch_size = drone_pos.shape[0]
        device = drone_pos.device
        max_targets = ordered_targets.shape[1]

        if current_idx is None:
            current_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Compute distance to current target.
        gather_idx = current_idx.clamp(0, max_targets - 1)
        current_target = ordered_targets[
            torch.arange(batch_size, device=device), gather_idx
        ]  # [batch, 3]
        dist = torch.norm(current_target - drone_pos, dim=-1)  # [batch]

        # Advance if within collection radius.
        advance = (dist < waypoint_collect_radius) & (current_idx < max_targets - 1)
        current_idx = current_idx + advance.long()

        # Recompute target after potential advance.
        gather_idx = current_idx.clamp(0, max_targets - 1)
        current_target = ordered_targets[
            torch.arange(batch_size, device=device), gather_idx
        ]
        done = current_idx >= max_targets - 1

        return current_target, current_idx, done

    def reset(self, batch_size: int, device: torch.device):
        """Reset internal state for new episodes."""
        self._current_target_idx = torch.zeros(batch_size, dtype=torch.long, device=device)


__all__ = ["WaypointSequencer"]
