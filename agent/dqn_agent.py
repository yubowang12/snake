"""DQN 智能体 —— 整合网络、经验回放、探索策略和优化器"""

import random
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.dqn_network import DQNNetwork
from agent.replay_buffer import ReplayBuffer
from agent.epsilon_scheduler import EpsilonScheduler


class DQNAgent:
    """Double DQN 智能体

    核心算法：
    1. 使用 online 网络选择动作
    2. 使用 target 网络评估 Q 值（Double DQN 解耦）
    3. 经验回放 + ε-贪心探索
    4. 梯度裁剪 + Huber Loss 稳定训练
    """

    def __init__(self, config):
        """
        Args:
            config: Config 数据类实例
        """
        self.config = config
        self.device = torch.device(config.device)
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.target_update_freq = config.target_update_freq
        self.use_double_dqn = config.use_double_dqn
        self.grad_clip_norm = config.grad_clip_norm

        # 初始化 online 和 target 网络
        self.online_net = DQNNetwork(
            input_channels=config.input_channels,
            grid_size=config.grid_size,
            n_actions=4,
        ).to(self.device)

        self.target_net = DQNNetwork(
            input_channels=config.input_channels,
            grid_size=config.grid_size,
            n_actions=4,
        ).to(self.device)

        # 同步 target 网络参数
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target 网络永远不训练

        # 优化器
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.lr)

        # Huber Loss (SmoothL1Loss) 比 MSE 更稳定
        self.loss_fn = nn.SmoothL1Loss()

        # 经验回放
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            state_shape=(config.input_channels, config.grid_size, config.grid_size),
        )

        # ε 调度器
        self.epsilon_scheduler = EpsilonScheduler(
            start=config.eps_start,
            end=config.eps_end,
            decay=config.eps_decay,
            mode='exponential',
        )

        self.steps_done = 0
        self.train_steps_done = 0

    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """根据 ε-贪心策略选择动作

        Args:
            state: 状态张量 (1, C, H, W)
            training: 训练模式（True 时使用 ε-贪心，False 时纯贪心）

        Returns:
            动作索引 0-3
        """
        if training and random.random() < self.epsilon_scheduler.get():
            return random.randint(0, 3)

        with torch.no_grad():
            q_values = self.online_net(state.to(self.device))
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def can_train(self) -> bool:
        """检查是否可以开始训练（缓冲区足够大）"""
        return len(self.replay_buffer) >= self.batch_size

    def train_step(self) -> float:
        """执行一步 DQN 训练（Double DQN 算法）

        Returns:
            损失值（若缓冲区不足则返回 0.0）
        """
        if not self.can_train():
            return 0.0

        # 采样一批经验
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size, self.device)

        # 当前状态的 Q(s, a)
        q_values = self.online_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: online 选动作, target 评价值
                next_actions = self.online_net(next_states).argmax(dim=1)
                next_q_target = self.target_net(next_states)
                next_q_value = next_q_target.gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Vanilla DQN: target 同时选动作和评价值
                next_q_target = self.target_net(next_states)
                next_q_value = next_q_target.max(dim=1).values

            target_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        # 计算损失并反向传播
        loss = self.loss_fn(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=self.grad_clip_norm
        )
        self.optimizer.step()

        self.train_steps_done += 1
        return float(loss.item())

    def update_target_network(self):
        """硬更新：将 online 权重复制到 target"""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def soft_update_target_network(self, tau: float = 0.005):
        """软更新：θ_target = τ * θ_online + (1 - τ) * θ_target"""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon_scheduler.step()

    def get_q_values(self, state: torch.Tensor) -> np.ndarray:
        """获取给定状态的所有 Q 值（用于分析）

        Args:
            state: 状态张量 (1, C, H, W)

        Returns:
            Q 值数组 shape (4,)
        """
        with torch.no_grad():
            q_values = self.online_net(state.to(self.device))
            return q_values.cpu().numpy().flatten()

    def save(self, path: str):
        """保存模型（含架构元数据）"""
        torch.save({
            'online_state_dict': self.online_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'grid_size': self.config.grid_size,
            'input_channels': self.config.input_channels,
            'n_actions': 4,
        }, path)

    def load(self, path: str):
        """加载模型，自动处理架构不匹配"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # 检查架构是否匹配
        saved_grid = checkpoint.get('grid_size', self.config.grid_size)
        saved_channels = checkpoint.get('input_channels', self.config.input_channels)

        need_rebuild = (saved_grid != self.config.grid_size or
                        saved_channels != self.config.input_channels)

        if need_rebuild:
            print(f"[警告] 检查点架构 ({saved_channels}ch, {saved_grid}x{saved_grid}) "
                  f"与当前配置 ({self.config.input_channels}ch, "
                  f"{self.config.grid_size}x{self.config.grid_size}) 不匹配，重建网络...")

            # 按保存的架构重建网络以加载权重
            from model.dqn_network import DQNNetwork
            self.online_net = DQNNetwork(
                input_channels=saved_channels,
                grid_size=saved_grid,
                n_actions=4,
            ).to(self.device)
            self.target_net = DQNNetwork(
                input_channels=saved_channels,
                grid_size=saved_grid,
                n_actions=4,
            ).to(self.device)

            # 重建优化器
            self.optimizer = optim.Adam(self.online_net.parameters(),
                                        lr=self.config.lr)

            # 重建经验回放缓冲区
            self.replay_buffer = ReplayBuffer(
                capacity=self.config.buffer_capacity,
                state_shape=(saved_channels, saved_grid, saved_grid),
            )

        self.online_net.load_state_dict(checkpoint['online_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.target_net.eval()

        # 尝试加载优化器状态
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            print("[警告] 优化器状态加载失败，使用新优化器")
