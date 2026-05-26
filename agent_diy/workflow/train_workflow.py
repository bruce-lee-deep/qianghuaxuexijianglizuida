#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Custom training workflow scaffold for Drone Obstacle Navigation.
无人机避障导航自定义训练工作流骨架。
"""


from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Training workflow entry for custom agents.
    自定义智能体训练工作流入口。
    """
    # The framework provides one active agent/env pair in this project.
    # 本项目中框架会提供一个有效的 agent/env 对。
    agent = agents[0]
    env = envs[0]

    # Load and validate the training configuration.
    # 加载并校验训练配置。
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        raise Exception("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")

    # Keep this workflow minimal; add rollout and optimization here if needed.
    # 当前工作流保持最小骨架；如需自定义训练，可在此补充采样与优化逻辑。
    logger.info("Custom DIY training workflow is active; extend rollout and optimization here if needed.")

    # Persist one model snapshot before exit.
    # 退出前保存一次模型快照。
    agent.save_model()

    env.close()

    return
