"""贪吃蛇游戏引擎 —— 纯逻辑，无渲染依赖，可在无 GUI 环境下高速运行"""

import random
from collections import deque
from typing import Tuple, Optional
import numpy as np

from utils.helpers import (
    DIRECTION_VECTORS,
    OPPOSITE_ACTION,
    ACTION_UP,
    ACTION_RIGHT,
    ACTION_DOWN,
    ACTION_LEFT,
    manhattan_distance,
)


class SnakeEngine:
    """贪吃蛇核心逻辑引擎

    在 grid_size × grid_size 的离散网格上运行。
    坐标系统: (row, col), (0,0) 为左上角。
    蛇体用双端队列存储，头部在索引 0。
    """

    def __init__(self, grid_size: int = 20, max_steps_without_food: int = 100,
                 reward_food: float = 10.0, reward_collision: float = -10.0,
                 reward_step: float = 0.0, reward_shaping: bool = False,
                 reward_shaping_scale: float = 0.5):
        self.grid_size = grid_size
        self.max_steps_without_food = max_steps_without_food
        self.reward_food = reward_food
        self.reward_collision = reward_collision
        self.reward_step = reward_step
        self.reward_shaping = reward_shaping
        self.reward_shaping_scale = reward_shaping_scale

        # 游戏状态
        self.snake: deque = deque()
        self.direction: int = ACTION_RIGHT
        self.food: Tuple[int, int] = (0, 0)
        self.score: int = 0
        self.steps: int = 0
        self.done: bool = False
        self.steps_since_food: int = 0

    def reset(self) -> np.ndarray:
        """重置游戏到初始状态，返回初始棋盘"""
        center = self.grid_size // 2
        # 蛇初始长度3，水平放置
        self.snake = deque([
            (center, center),       # 头
            (center, center - 1),   # 身
            (center, center - 2),   # 尾
        ])
        self.direction = ACTION_RIGHT
        self.score = 0
        self.steps = 0
        self.done = False
        self.steps_since_food = 0
        self._place_food()
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """执行一步动作，返回 (next_state, reward, done)

        Args:
            action: 0=上, 1=右, 2=下, 3=左

        Returns:
            (board_state, reward, done)
        """
        if self.done:
            return self.get_state(), 0.0, True

        # 1. 更新方向（防止180度掉头）
        if not self._is_opposite(action, self.direction):
            self.direction = action

        # 2. 计算新头部位置
        dr, dc = DIRECTION_VECTORS[self.direction]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)
        new_r, new_c = new_head

        # 3. 塑形奖励（移动前距离 vs 移动后距离）
        shaping_reward = 0.0
        if self.reward_shaping:
            prev_dist = manhattan_distance(head_r, head_c, *self.food)
            new_dist = manhattan_distance(new_r, new_c, *self.food)
            shaping_reward = (prev_dist - new_dist) * self.reward_shaping_scale

        # 4. 边界碰撞检测
        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            self.done = True
            return self.get_state(), self.reward_collision, True

        # 5. 自身碰撞检测（注意：尾部会在移动后消失，所以 tail 不算碰撞）
        # 但如果蛇吃到食物，尾部不会消失，所以需要分别处理
        will_eat = (new_head == self.food)

        # 检查是否撞到自己身体（排除即将消失的尾部尾尖）
        body_to_check = list(self.snake)
        if not will_eat:
            body_to_check = body_to_check[:-1]  # 尾部会消失，排除尾尖

        if new_head in body_to_check:
            self.done = True
            return self.get_state(), self.reward_collision, True

        # 6. 移动蛇头
        self.snake.appendleft(new_head)

        # 7. 处理食物
        if will_eat:
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
            # 吃到食物时不 pop 尾部 → 蛇增长
            reward = self.reward_food + shaping_reward
        else:
            self.snake.pop()  # 正常移动，尾部消失
            self.steps_since_food += 1
            reward = self.reward_step + shaping_reward

        # 8. 饥饿超时
        if self.steps_since_food >= self.max_steps_without_food:
            self.done = True

        self.steps += 1
        return self.get_state(), reward, self.done

    def get_state(self) -> np.ndarray:
        """返回当前棋盘状态

        Returns:
            np.ndarray shape (grid_size, grid_size)
            0=空地, 1=蛇身, 2=蛇头, 3=食物
        """
        board = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # 蛇身
        for i, (r, c) in enumerate(self.snake):
            if i == 0:
                board[r, c] = 2  # 头
            else:
                board[r, c] = 1  # 身

        # 食物
        fr, fc = self.food
        board[fr, fc] = 3

        return board

    def _place_food(self):
        """在空白位置随机放置食物"""
        empty_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.snake
        ]
        if not empty_cells:
            self.done = True  # 蛇填满整个棋盘 → 胜利
            return
        self.food = random.choice(empty_cells)

    @staticmethod
    def _is_opposite(action: int, current_direction: int) -> bool:
        """判断 action 是否为当前方向的 180° 反向"""
        return action == OPPOSITE_ACTION.get(current_direction, -1)
