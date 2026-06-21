"""经验回放缓冲区 —— 存储和采样经验元组"""

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """循环经验回放缓冲区

    存储 (state, action, reward, next_state, done) 五元组。
    使用预分配的 numpy 数组提升内存效率。
    """

    def __init__(self, capacity: int = 100_000, state_shape: Tuple[int, ...] = (4, 20, 20)):
        """
        Args:
            capacity: 最大存储容量
            state_shape: 单个状态的形状 (channels, H, W)
        """
        self.capacity = capacity
        self.state_shape = state_shape

        # 预分配内存
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.pos = 0       # 当前写入位置
        self.size = 0      # 当前存储数量
        self.full = False  # 是否已填满

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """存储一条经验

        Args:
            state: 当前状态 (C, H, W)
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态 (C, H, W)
            done: 是否终止
        """
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.full = True

    def sample(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
        """均匀随机采样一批经验

        Args:
            batch_size: 批量大小
            device: 目标设备

        Returns:
            (states, actions, rewards, next_states, dones) — 均为 torch.Tensor
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.from_numpy(self.states[indices]).to(device),
            torch.from_numpy(self.actions[indices]).to(device),
            torch.from_numpy(self.rewards[indices]).to(device),
            torch.from_numpy(self.next_states[indices]).to(device),
            torch.from_numpy(self.dones[indices]).to(device),
        )

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """缓冲区是否达到最小采样要求"""
        return self.size >= min_size
