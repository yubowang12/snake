"""状态编码器：将游戏棋盘转换为多通道张量"""

import numpy as np
import torch
from game.engine import SnakeEngine


class StateEncoder:
    """将 SnakeEngine 的状态编码为适合 CNN 输入的多通道张量

    输出形状: (4, grid_size, grid_size)，4个通道分别为：
      Channel 0: 蛇身（所有段）
      Channel 1: 蛇头
      Channel 2: 食物
      Channel 3: 危险掩码（边界 + 蛇身除尾尖）
    """

    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size

    def encode(self, engine: SnakeEngine) -> torch.Tensor:
        """将引擎状态编码为张量

        Args:
            engine: 游戏引擎实例

        Returns:
            torch.Tensor shape (1, 4, grid_size, grid_size) 含 batch 维度
        """
        state = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: 蛇身（全部段，包括头）
        for r, c in engine.snake:
            state[0, r, c] = 1.0

        # Channel 1: 蛇头（单独标记）
        if engine.snake:
            head_r, head_c = engine.snake[0]
            state[1, head_r, head_c] = 1.0

        # Channel 2: 食物
        food_r, food_c = engine.food
        state[2, food_r, food_c] = 1.0

        # Channel 3: 危险区域（边界 + 蛇身除尾尖）
        # 边界
        state[3, 0, :] = 1.0
        state[3, -1, :] = 1.0
        state[3, :, 0] = 1.0
        state[3, :, -1] = 1.0

        # 蛇身（排除尾尖，因为移动时尾部会消失）
        snake_list = list(engine.snake)
        for r, c in snake_list[:-1]:
            state[3, r, c] = 1.0

        return torch.from_numpy(state).unsqueeze(0)  # (1, 4, H, W)

    def encode_batch(self, states: np.ndarray) -> torch.Tensor:
        """批量编码多个原始状态

        Args:
            states: np.ndarray shape (B, grid_size, grid_size)

        Returns:
            torch.Tensor shape (B, 4, grid_size, grid_size)
        """
        B = states.shape[0]
        encoded = np.zeros((B, 4, self.grid_size, self.grid_size), dtype=np.float32)

        for b in range(B):
            board = states[b]
            # Channel 0: 蛇身 (value 1 or 2)
            encoded[b, 0] = (board >= 1) & (board <= 2)
            # Channel 1: 蛇头 (value 2)
            encoded[b, 1] = (board == 2)
            # Channel 2: 食物 (value 3)
            encoded[b, 2] = (board == 3)
            # Channel 3: 危险区域 - 边界
            encoded[b, 3, 0, :] = 1.0
            encoded[b, 3, -1, :] = 1.0
            encoded[b, 3, :, 0] = 1.0
            encoded[b, 3, :, -1] = 1.0
            # Channel 3: 危险区域 - 蛇身 (value 1, 不包括头 value 2)
            encoded[b, 3] += (board == 1)

        return torch.from_numpy(encoded)
