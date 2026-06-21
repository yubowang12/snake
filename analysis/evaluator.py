"""评估器 —— 对训练好的模型进行离线评估和基准测试"""

import sys
import os
import time
import random
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import Config
from game.engine import SnakeEngine
from game.renderer import SnakeRenderer
from model.state_encoder import StateEncoder
from agent.dqn_agent import DQNAgent


class Evaluator:
    """模型评估器

    功能：
    1. 加载训练好的模型进行离线评估
    2. 运行随机策略作为基线对比
    3. 演示模式（带渲染的单局游戏）
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        self.engine = SnakeEngine(
            grid_size=config.grid_size,
            max_steps_without_food=config.max_steps_without_food,
            reward_food=config.reward_food,
            reward_collision=config.reward_collision,
        )
        self.encoder = StateEncoder(grid_size=config.grid_size)
        self.agent = DQNAgent(config)
        self.agent.online_net.eval()  # 评估模式

    def load_model(self, checkpoint_path: str):
        """加载训练好的模型"""
        self.agent.load(checkpoint_path)
        self.agent.online_net.eval()
        print(f"[评估] 模型已加载: {checkpoint_path}")

    def evaluate(self, n_episodes: int = 100, render: bool = False) -> Dict:
        """运行评估 episode

        Args:
            n_episodes: 评估 episode 数量
            render: 是否渲染

        Returns:
            统计字典: mean_score, max_score, min_score, std_score, mean_steps, mean_reward
        """
        scores = []
        total_rewards = []
        steps_survived = []

        renderer = None
        if render:
            renderer = SnakeRenderer(self.engine, fps=10)

        print(f"\n[评估] 运行 {n_episodes} 局 (ε=0, 纯剥削)...")

        for ep in range(n_episodes):
            _ = self.engine.reset()
            state = self.encoder.encode(self.engine)
            episode_reward = 0.0
            steps = 0

            while True:
                action = self.agent.select_action(state, training=False)
                _, reward, done = self.engine.step(action)
                state = self.encoder.encode(self.engine)
                episode_reward += reward
                steps += 1

                if renderer:
                    renderer.render()

                if done:
                    break

            scores.append(self.engine.score)
            total_rewards.append(episode_reward)
            steps_survived.append(steps)

            if (ep + 1) % 20 == 0:
                print(f"  已完成 {ep + 1}/{n_episodes} 局...")

        if renderer:
            renderer.close()

        stats = {
            'mean_score': float(np.mean(scores)),
            'max_score': int(np.max(scores)),
            'min_score': int(np.min(scores)),
            'std_score': float(np.std(scores)),
            'mean_steps': float(np.mean(steps_survived)),
            'mean_reward': float(np.mean(total_rewards)),
            'scores': scores,
            'steps': steps_survived,
        }

        self._print_stats(stats)
        return stats

    def demo(self, checkpoint_path: Optional[str] = None):
        """演示模式：带渲染的单局游戏

        Args:
            checkpoint_path: 模型路径（若已加载可跳过）
        """
        if checkpoint_path:
            self.load_model(checkpoint_path)

        renderer = SnakeRenderer(self.engine, cell_size=30, fps=10)
        config_demo = Config()
        config_demo.render = True

        print("\n[演示] 开始单局游戏演示...")
        _ = self.engine.reset()
        state = self.encoder.encode(self.engine)

        while True:
            action = self.agent.select_action(state, training=False)
            _, reward, done = self.engine.step(action)
            state = self.encoder.encode(self.engine)

            if not renderer.render():
                break

            if done:
                print(f"游戏结束! 分数: {self.engine.score}, 步数: {self.engine.steps}")
                # 等待一会儿再关闭
                time.sleep(2)
                break

        renderer.close()

    @staticmethod
    def evaluate_random(n_episodes: int = 100, grid_size: int = 20) -> Dict:
        """随机策略基线评估

        Args:
            n_episodes: 评估 episode 数量
            grid_size: 网格大小

        Returns:
            统计字典
        """
        engine = SnakeEngine(grid_size=grid_size)
        scores = []
        steps_survived = []

        print(f"\n[随机基线] 运行 {n_episodes} 局随机策略...")

        for ep in range(n_episodes):
            _ = engine.reset()
            steps = 0

            while True:
                action = random.randint(0, 3)
                _, _, done = engine.step(action)
                steps += 1
                if done:
                    break

            scores.append(engine.score)
            steps_survived.append(steps)

        stats = {
            'mean_score': float(np.mean(scores)),
            'max_score': int(np.max(scores)),
            'std_score': float(np.std(scores)),
            'mean_steps': float(np.mean(steps_survived)),
        }

        print(f"\n[随机基线结果]")
        print(f"  平均分数: {stats['mean_score']:.2f}")
        print(f"  最大分数: {stats['max_score']}")
        print(f"  平均步数: {stats['mean_steps']:.1f}")

        return stats

    def _print_stats(self, stats: Dict):
        """打印评估统计结果"""
        print(f"\n{'='*50}")
        print(f"评估结果 ({len(stats['scores'])} 局)")
        print(f"{'='*50}")
        print(f"  平均分数:    {stats['mean_score']:.2f}")
        print(f"  最大分数:    {stats['max_score']}")
        print(f"  最小分数:    {stats['min_score']}")
        print(f"  标准差:      {stats['std_score']:.2f}")
        print(f"  平均步数:    {stats['mean_steps']:.1f}")
        print(f"  平均奖励:    {stats['mean_reward']:.2f}")
        print(f"{'='*50}")
