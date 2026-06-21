"""Pygame 可视化渲染器"""

import pygame
import numpy as np
from game.engine import SnakeEngine


# 颜色定义
COLOR_BG = (30, 30, 30)           # 背景
COLOR_GRID = (50, 50, 50)         # 网格线
COLOR_SNAKE_BODY = (0, 200, 80)   # 蛇身绿色
COLOR_SNAKE_HEAD = (0, 255, 120)  # 蛇头亮绿色
COLOR_FOOD = (255, 60, 60)        # 食物红色
COLOR_TEXT = (220, 220, 220)      # 文字


class SnakeRenderer:
    """Pygame 贪吃蛇渲染器

    将 SnakeEngine 的游戏状态可视化渲染。
    仅在训练时需要可视化时使用，否则关闭以加速训练。
    """

    def __init__(self, engine: SnakeEngine, cell_size: int = 30, fps: int = 10):
        self.engine = engine
        self.cell_size = cell_size
        self.fps = fps
        self.width = engine.grid_size * cell_size
        self.height = engine.grid_size * cell_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height + 40))
        pygame.display.set_caption("贪吃蛇 DQN 深度强化学习")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('microsoftyahei', 18)

    def render(self):
        """渲染一帧"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(COLOR_BG)

        # 绘制网格线
        for i in range(self.engine.grid_size + 1):
            pos = i * self.cell_size
            pygame.draw.line(self.screen, COLOR_GRID, (pos, 0), (pos, self.height))
            pygame.draw.line(self.screen, COLOR_GRID, (0, pos), (self.width, pos))

        # 绘制食物
        fr, fc = self.engine.food
        food_center = (
            fc * self.cell_size + self.cell_size // 2,
            fr * self.cell_size + self.cell_size // 2,
        )
        pygame.draw.circle(
            self.screen, COLOR_FOOD, food_center,
            self.cell_size // 2 - 2
        )

        # 绘制蛇
        for i, (r, c) in enumerate(self.engine.snake):
            x = c * self.cell_size + 1
            y = r * self.cell_size + 1
            size = self.cell_size - 2
            if i == 0:
                color = COLOR_SNAKE_HEAD
            else:
                color = COLOR_SNAKE_BODY
            pygame.draw.rect(self.screen, color, (x, y, size, size), border_radius=4)

        # 绘制信息栏
        info_text = (
            f"分数: {self.engine.score}  |  "
            f"步数: {self.engine.steps}  |  "
            f"长度: {len(self.engine.snake)}"
        )
        text_surface = self.font.render(info_text, True, COLOR_TEXT)
        self.screen.blit(text_surface, (10, self.height + 10))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self):
        """关闭渲染窗口"""
        pygame.quit()
