#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration for Drone Obstacle Navigation.
无人机避障导航监控面板配置。
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder
from tools.cluster_monitor import (
    CLUSTER_OBSTACLE_NUMS,
    CLUSTER_OBSTACLE_RADII,
    CLUSTER_WAYPOINT_COUNTS,
    build_obstacle_num_metric_key,
    build_obstacle_radius_metric_key,
    build_waypoint_count_metric_key,
    radius_to_tag,
)


def _obstacle_num_series(metric_prefix):
    return [
        (
            f"n{obstacle_num}",
            f"avg({build_obstacle_num_metric_key(metric_prefix, obstacle_num)}{{}})",
        )
        for obstacle_num in CLUSTER_OBSTACLE_NUMS
    ]


def _obstacle_radius_series(metric_prefix):
    return [
        (
            f"r{radius_to_tag(obstacle_radius)}",
            f"avg({build_obstacle_radius_metric_key(metric_prefix, obstacle_radius)}{{}})",
        )
        for obstacle_radius in CLUSTER_OBSTACLE_RADII
    ]


def _waypoint_count_series(metric_prefix):
    return [
        (
            f"w{waypoint_count}",
            f"avg({build_waypoint_count_metric_key(metric_prefix, waypoint_count)}{{}})",
        )
        for waypoint_count in CLUSTER_WAYPOINT_COUNTS
    ]


def _add_metric_series(builder, metric_pairs):
    for metrics_name, expr in metric_pairs:
        builder.add_metric(metrics_name=metrics_name, expr=expr)
    return builder


def build_monitor():
    """Create monitor panel configurations."""
    monitor = MonitorConfigBuilder()

    config = (
        monitor.title("无人机")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="累积回报", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .add_panel(name="总损失", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .end_panel()
        .add_panel(name="熵损失", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .end_group()
        .add_group(group_name="任务总览", group_name_en="task_overview")
        .add_panel(name="任务结果", name_en="task_funnel", type="line")
        .add_metric(metrics_name="arrival_rate", expr="avg(arrival_rate{})")
        .add_metric(metrics_name="success", expr="avg(success{})")
        .add_metric(metrics_name="failed", expr="avg(failed{})")
        .add_metric(metrics_name="timeout", expr="avg(timeout{})")
        .end_panel()
        .add_panel(name="总分拆解", name_en="score_breakdown", type="line")
        .add_metric(metrics_name="total_score", expr="avg(total_score{})")
        .add_metric(metrics_name="nav_score", expr="avg(nav_score{})")
        .add_metric(metrics_name="hover_score", expr="avg(hover_score{})")
        .add_metric(metrics_name="waypoint_score", expr="avg(waypoint_score{})")
        .end_panel()
        .add_panel(name="导航子项", name_en="nav_components", type="line")
        .add_metric(metrics_name="time_norm", expr="avg(time_norm{})")
        .add_metric(metrics_name="smooth_norm", expr="avg(smooth_norm{})")
        .end_panel()
        .add_panel(name="过程质量", name_en="process_quality", type="line")
        .add_metric(metrics_name="collision_count", expr="avg(collision_count{})")
        .add_metric(metrics_name="max_collisions", expr="avg(max_collisions{})")
        .add_metric(metrics_name="hover_precision", expr="avg(hover_precision{})")
        .end_panel()
        .end_group()
        .add_group(group_name="障碍数量", group_name_en="obstacle_num")
        .add_panel(name="到达成功率", name_en="arrival_rate_by_obstacle_num", type="line")
    )

    _add_metric_series(config, _obstacle_num_series("arrival_rate"))
    config = config.end_panel().add_panel(name="平均完成步数", name_en="completion_cost_by_obstacle_num", type="line")
    _add_metric_series(config, _obstacle_num_series("completion_cost"))
    config = config.end_panel().add_panel(name="碰撞次数", name_en="collision_count_by_obstacle_num", type="line")
    _add_metric_series(config, _obstacle_num_series("collision_count"))

    config = (
        config.end_panel()
        .end_group()
        .add_group(group_name="障碍半径", group_name_en="obstacle_radius")
        .add_panel(name="到达成功率", name_en="arrival_rate_by_obstacle_radius", type="line")
    )

    _add_metric_series(config, _obstacle_radius_series("arrival_rate"))
    config = config.end_panel().add_panel(
        name="平均完成步数", name_en="completion_cost_by_obstacle_radius", type="line"
    )
    _add_metric_series(config, _obstacle_radius_series("completion_cost"))
    config = config.end_panel().add_panel(name="碰撞次数", name_en="collision_count_by_obstacle_radius", type="line")
    _add_metric_series(config, _obstacle_radius_series("collision_count"))

    config = (
        config.end_panel()
        .end_group()
        .add_group(group_name="途径点数量", group_name_en="waypoint_count")
        .add_panel(name="平均总分", name_en="total_score_by_waypoint_count", type="line")
    )

    _add_metric_series(config, _waypoint_count_series("total_score"))
    config = config.end_panel().add_panel(name="平均完成步数", name_en="completion_cost_by_waypoint_count", type="line")
    _add_metric_series(config, _waypoint_count_series("completion_cost"))
    config = config.end_panel().add_panel(
        name="获得途径点数量", name_en="waypoints_visited_by_waypoint_count", type="line"
    )
    _add_metric_series(config, _waypoint_count_series("waypoints_visited"))

    return config.end_panel().end_group().build()
