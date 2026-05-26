#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
最少航路点规划器。

在 3D 空间中规划从起点到终点的最少航路点序列，
使得每段平飞（固定高度直线飞行）不穿过任何障碍物。

算法：可见性图 + BFS 最短路径 + 贪心剪枝
"""

from __future__ import annotations
import math
import torch
from agent_ppo.conf.conf import Config


class WaypointPlanner:
    """规划从 start 到 goal 的最少避障航路点。

    坐标系说明
    ----------
    所有输入均在无人机的相对坐标系中（无人机 = 原点）。
    障碍物/终点位置在世界中是固定的，因此：
        wp_offset = wp_rpos_at_plan - goal_rpos_at_plan   (世界常数)
        wp_rpos(t) = goal_rpos(t) + wp_offset              (任意时刻)

    航路点与 waypoint 的区别
    ------------------------
    - 航路点：纯空间位置 (x,y,z)，用于避开障碍物的导航锚点
    - 环境 waypoint：观测中的途经点，是可选的得分目标
    - 航路点不占用观测槽位，仅作为飞行控制的中间目标
    """

    def __init__(self):
        cfg = Config
        self._safety_radius = cfg.SAFETY_RADIUS       # 25cm 安全距离
        self._drone_radius = 0.05                      # 无人机半径
        self._arena_bounds = {
            "x_min": cfg.ARENA_X_MIN, "x_max": cfg.ARENA_X_MAX,
            "y_min": cfg.ARENA_Y_MIN, "y_max": cfg.ARENA_Y_MAX,
            "z_min": cfg.ARENA_Z_MIN, "z_max": cfg.ARENA_Z_MAX,
        }
        self._num_directions = 8  # 围绕障碍物生成的候选方向数

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def plan(
        self,
        start_rpos: torch.Tensor,        # [3]
        goal_rpos: torch.Tensor,          # [3]
        obstacle_rpos: torch.Tensor,      # [max_obs, 3]
        obstacle_radius: torch.Tensor,    # [max_obs]
        obstacle_active: torch.Tensor,    # [max_obs] bool
    ):
        """规划最少航路点。

        Returns
        -------
        waypoints: list of [x, y, z]     在规划时刻无人机相对系中的航路点（可能为空）
        offsets:   list of [x, y, z]     每个航路点相对于终点的偏移（世界常数）
        total:     int                   总航路点数（含终点）
        """
        start = start_rpos.detach().cpu().tolist()
        goal = goal_rpos.detach().cpu().tolist()

        # 提取活跃障碍物
        active_mask = obstacle_active.detach().cpu()
        obs_pos = obstacle_rpos.detach().cpu()
        obs_rad = obstacle_radius.detach().cpu()

        obstacles = []
        for i in range(len(active_mask)):
            if active_mask[i]:
                pos = obs_pos[i].tolist()
                r = obs_rad[i].item()
                if r > 1e-6:
                    obstacles.append({"pos": pos, "radius": r})

        # 1. 检查直达路径
        if self._direct_path_clear(start, goal, obstacles):
            return [], [], 1  # 0 个航路点，共 1 个目标（终点）

        # 2. 生成候选航路点
        candidates = self._generate_candidates(start, goal, obstacles)

        # 3. 构建可见性图
        graph, nodes = self._build_graph(start, goal, candidates, obstacles)

        if not graph:
            # 图为空 → 回溯到直接飞（即使有碰撞风险）
            return [], [], 1

        # 4. BFS 找最短路径（最少节点数）
        start_idx = 0   # nodes[0] == start
        goal_idx = 1    # nodes[1] == goal
        path_indices = self._bfs_shortest_path(graph, start_idx, goal_idx)

        if path_indices is None:
            return [], [], 1

        # 5. 贪心剪枝
        pruned = self._prune_path(path_indices, nodes, obstacles)

        # 提取航路点（去掉起点和终点）
        waypoints = []
        for idx in pruned:
            if idx == start_idx or idx == goal_idx:
                continue
            waypoints.append(nodes[idx])

        # 计算每个航路点相对于终点的偏移（世界常数）
        offsets = []
        for wp in waypoints:
            offsets.append([
                wp[0] - goal[0],
                wp[1] - goal[1],
                wp[2] - goal[2],
            ])

        total = len(waypoints) + 1  # 航路点 + 终点
        return waypoints, offsets, total

    def waypoint_rpos_now(
        self,
        goal_rpos_now: torch.Tensor,  # [3]
        offsets: list,                 # list of [x, y, z] offsets from goal
    ):
        """根据当前的 goal_rpos 和偏移量，计算航路点的当前 rpos。

        wp_world = goal_world + offset
        wp_rpos(t) = wp_world - drone(t)
                   = (goal_world + offset) - drone(t)
                   = goal_rpos(t) + offset
        """
        goal = goal_rpos_now.detach().cpu().tolist()
        result = []
        for off in offsets:
            result.append([
                goal[0] + off[0],
                goal[1] + off[1],
                goal[2] + off[2],
            ])
        return result

    # ------------------------------------------------------------------
    # 碰撞检测
    # ------------------------------------------------------------------

    def _segment_clear(self, a_xy, b_xy, flight_z, obstacles):
        """检查从 a_xy 到 b_xy、高度 flight_z 的平飞段是否无碰撞。"""
        for obs in obstacles:
            oz = obs["pos"][2]
            r = obs["radius"]

            # 垂直方向：飞行高度是否在障碍物垂直范围内
            vert_overlap = abs(flight_z - oz) < (r + self._drone_radius)

            if not vert_overlap:
                continue  # 从上方或下方飞过，无碰撞

            # 水平方向：检查点到线段距离
            d = self._point_to_segment(obs["pos"][:2], a_xy, b_xy)
            if d < r + self._safety_radius:
                return False
        return True

    @staticmethod
    def _point_to_segment(p, a, b):
        """点 p 到线段 ab 的最短距离（2D）。"""
        px, py = p
        ax, ay = a
        bx, by = b

        abx = bx - ax
        aby = by - ay
        if abs(abx) < 1e-9 and abs(aby) < 1e-9:
            # a ≈ b
            dx = px - ax
            dy = py - ay
            return math.sqrt(dx * dx + dy * dy)

        # 投影参数 t
        t = ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)
        t = max(0.0, min(1.0, t))

        # 垂足
        proj_x = ax + t * abx
        proj_y = ay + t * aby

        dx = px - proj_x
        dy = py - proj_y
        return math.sqrt(dx * dx + dy * dy)

    def _direct_path_clear(self, start, goal, obstacles):
        """检查 start→goal 的直接平飞路径是否通畅。"""
        flight_z = goal[2]  # 平飞高度 = 终点高度
        return self._segment_clear(start[:2], goal[:2], flight_z, obstacles)

    # ------------------------------------------------------------------
    # 候选生成
    # ------------------------------------------------------------------

    def _generate_candidates(self, start, goal, obstacles):
        """为每个阻塞路径的障碍物生成绕行候选点。"""
        candidates = []
        seen = set()

        direct_blockers = self._find_blockers(start, goal, obstacles)

        for obs in obstacles:
            # 只为直接路径上的障碍物生成候选
            if obs not in direct_blockers:
                # 但仍然可能成为后续段的障碍，所以也生成
                pass

            ox, oy, oz = obs["pos"]
            r = obs["radius"]
            safe_dist = r + self._safety_radius + self._drone_radius

            for i in range(self._num_directions):
                angle = 2.0 * math.pi * i / self._num_directions
                cx = ox + safe_dist * math.cos(angle)
                cy = oy + safe_dist * math.sin(angle)

                # 夹紧到场地内
                b = self._arena_bounds
                cx = max(b["x_min"] + 0.2, min(b["x_max"] - 0.2, cx))
                cy = max(b["y_min"] + 0.2, min(b["y_max"] - 0.2, cy))

                # 航路点高度固定为终点高度（先爬升到位，再水平绕行）
                cz = goal[2]

                key = (round(cx, 3), round(cy, 3), round(cz, 3))
                if key in seen:
                    continue

                # 验证候选不在任何障碍物内
                if self._point_in_obstacle((cx, cy, cz), obstacles):
                    continue

                seen.add(key)
                candidates.append([cx, cy, cz])

        return candidates

    def _find_blockers(self, start, goal, obstacles):
        """找出直接阻挡 start→goal 路径的障碍物。"""
        blockers = []
        flight_z = goal[2]
        for obs in obstacles:
            oz = obs["pos"][2]
            r = obs["radius"]
            if abs(flight_z - oz) < r + self._drone_radius:
                d = self._point_to_segment(obs["pos"][:2], start[:2], goal[:2])
                if d < r + self._safety_radius:
                    blockers.append(obs)
        return blockers

    def _point_in_obstacle(self, point, obstacles):
        """检查点是否在任何障碍物内部。"""
        px, py, pz = point
        for obs in obstacles:
            ox, oy, oz = obs["pos"]
            r = obs["radius"]
            # 3D 距离
            d = math.sqrt((px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2)
            if d < r + self._drone_radius:
                return True
        return False

    # ------------------------------------------------------------------
    # 可见性图
    # ------------------------------------------------------------------

    def _build_graph(self, start, goal, candidates, obstacles):
        """构建可见性图。

        节点: [start, goal, candidate_0, candidate_1, ...]
        边:   A→B 存在 当且仅当 level flight at z_B from A_xy to B_xy 无碰撞
        """
        nodes = [start, goal] + candidates
        n = len(nodes)
        graph = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a = nodes[i]
                b = nodes[j]
                # 平飞高度 = 目标点的高度
                flight_z = b[2]
                if self._segment_clear(a[:2], b[:2], flight_z, obstacles):
                    graph[i].append(j)

        return graph, nodes

    # ------------------------------------------------------------------
    # 最短路径
    # ------------------------------------------------------------------

    def _bfs_shortest_path(self, graph, start_idx, goal_idx):
        """BFS 寻找节点数最少的路径（所有边权重为 1）。"""
        from collections import deque

        visited = {start_idx}
        parent = {start_idx: None}
        queue = deque([start_idx])

        while queue:
            current = queue.popleft()

            if current == goal_idx:
                # 回溯重建路径
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path

            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return None  # 无路径

    # ------------------------------------------------------------------
    # 贪心剪枝
    # ------------------------------------------------------------------

    def _prune_path(self, path_indices, nodes, obstacles):
        """贪心剪枝：如果跳过中间点仍然通畅，就去掉它。"""
        if len(path_indices) <= 2:
            return path_indices

        pruned = list(path_indices)
        changed = True

        while changed:
            changed = False
            i = 0
            while i < len(pruned) - 2:
                a_idx = pruned[i]
                c_idx = pruned[i + 2]
                a = nodes[a_idx]
                c = nodes[c_idx]
                # 从 a 直接飞到 c（平飞高度 = c.z）
                if self._segment_clear(a[:2], c[:2], c[2], obstacles):
                    # 跳过 b
                    pruned.pop(i + 1)
                    changed = True
                else:
                    i += 1

        return pruned


__all__ = ["WaypointPlanner"]
