"""训练循环 —— 编排游戏引擎与 DQN 智能体的交互"""

import time
import sys
import os
from collections import defaultdict
import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from game.engine import SnakeEngine
from game.renderer import SnakeRenderer
from model.state_encoder import StateEncoder
from agent.dqn_agent import DQNAgent
from training.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """DQN 训练器

    编排训练循环：
    1. 初始化环境和智能体
    2. 每个 episode：重置 → 交互 → 存储经验 → 训练 → 记录指标
    3. 定期保存检查点和更新目标网络
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        # 初始化各组件
        self.engine = SnakeEngine(
            grid_size=config.grid_size,
            max_steps_without_food=config.max_steps_without_food,
            reward_food=config.reward_food,
            reward_collision=config.reward_collision,
            reward_step=config.reward_step,
            reward_shaping=config.reward_shaping,
            reward_shaping_scale=config.reward_shaping_scale,
        )
        self.encoder = StateEncoder(grid_size=config.grid_size)
        self.agent = DQNAgent(config)

        # 渲染器（仅需要时创建）
        self.renderer = None
        if config.render:
            self.renderer = SnakeRenderer(
                self.engine,
                cell_size=config.cell_size,
                fps=config.render_fps,
            )

        # 训练指标
        self.metrics = defaultdict(list)
        self.start_episode = 0

        print(f"[初始化] 设备: {config.device}")
        print(f"[初始化] 网格大小: {config.grid_size}×{config.grid_size}")
        print(f"[初始化] 经验回放容量: {config.buffer_capacity}")
        print(f"[初始化] Double DQN: {config.use_double_dqn}")
        print(f"[初始化] ε衰减: {config.eps_start} → {config.eps_end} (×{config.eps_decay}/ep)")

    def train(self, n_episodes: int = None):
        """主训练循环

        Args:
            n_episodes: 训练 episode 数（默认使用 config 中的设置）
        """
        if n_episodes is None:
            n_episodes = self.config.n_episodes

        print(f"\n{'='*60}")
        print(f"开始训练 | 共 {n_episodes} episodes")
        print(f"按 Ctrl+C 可随时停止（会自动保存模型）")
        print(f"{'='*60}\n")

        best_avg_score = 0.0
        episode = self.start_episode

        try:
            for episode in range(self.start_episode, self.start_episode + n_episodes):
                # 重置环境
                _ = self.engine.reset()
                state = self.encoder.encode(self.engine)

                episode_reward = 0.0
                episode_loss = 0.0
                episode_steps = 0

                while True:
                    # 1. 选择动作
                    action = self.agent.select_action(state, training=True)

                    # 2. 执行动作
                    _, reward, done = self.engine.step(action)
                    next_state = self.encoder.encode(self.engine)

                    # 3. 存储经验
                    self.agent.store_transition(
                        state.squeeze(0).numpy(),
                        action,
                        reward,
                        next_state.squeeze(0).numpy(),
                        done,
                    )

                    # 4. 训练一步
                    loss = self.agent.train_step()
                    episode_loss += loss
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state

                    # 5. 渲染（如启用）
                    if self.renderer:
                        if not self.renderer.render():
                            self._cleanup()
                            return

                    if done:
                        break

                # Episode 结束处理
                self.agent.decay_epsilon()

                # 更新目标网络
                if (episode + 1) % self.config.target_update_freq == 0:
                    self.agent.update_target_network()

                # 记录指标
                self.metrics['episode_reward'].append(episode_reward)
                self.metrics['episode_steps'].append(episode_steps)
                self.metrics['avg_loss'].append(episode_loss / max(1, episode_steps))
                self.metrics['epsilon'].append(self.agent.epsilon_scheduler.get())
                self.metrics['score'].append(self.engine.score)
                self.metrics['snake_length'].append(len(self.engine.snake))

                # 定期打印
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(self.metrics['episode_reward'][-100:])
                    avg_score = np.mean(self.metrics['score'][-100:])
                    max_score = max(self.metrics['score'][-100:])
                    eps = self.agent.epsilon_scheduler.get()

                    print(
                        f"Ep {episode + 1:5d}/{self.start_episode + n_episodes} | "
                        f"Avg奖励(100): {avg_reward:7.2f} | "
                        f"Avg分数: {avg_score:5.1f} | "
                        f"最大分数: {max_score:3d} | "
                        f"ε: {eps:.4f} | "
                        f"步数: {episode_steps:4d}"
                    )

                    # 更新最佳平均分数
                    if avg_score > best_avg_score:
                        best_avg_score = avg_score
                        best_path = os.path.join(self.config.checkpoint_dir, 'best.pth')
                        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                        torch.save({
                            'episode': episode + 1,
                            'online_state_dict': self.agent.online_net.state_dict(),
                            'target_state_dict': self.agent.target_net.state_dict(),
                            'optimizer_state_dict': self.agent.optimizer.state_dict(),
                            'metrics': dict(self.metrics),
                            'epsilon': eps,
                            'grid_size': self.config.grid_size,
                            'input_channels': self.config.input_channels,
                            'n_actions': 4,
                        }, best_path)
                        print(f"  >>> 新最佳模型! Avg分数={avg_score:.1f}")

                # 定期保存检查点
                if (episode + 1) % self.config.checkpoint_freq == 0:
                    save_checkpoint(
                        self.agent, dict(self.metrics),
                        episode + 1, self.config.checkpoint_dir,
                    )

        except KeyboardInterrupt:
            print(f"\n\n[中断] 用户按了 Ctrl+C，正在保存模型...")
            # 保存中断时的模型
            interrupt_path = os.path.join(self.config.checkpoint_dir, 'interrupt.pth')
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            torch.save({
                'episode': episode + 1,
                'online_state_dict': self.agent.online_net.state_dict(),
                'target_state_dict': self.agent.target_net.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'metrics': dict(self.metrics),
                'epsilon': self.agent.epsilon_scheduler.get(),
                'grid_size': self.config.grid_size,
                'input_channels': self.config.input_channels,
                'n_actions': 4,
            }, interrupt_path)
            print(f"[中断] 模型已保存到 {interrupt_path}")
            print(f"[中断] 续训命令: python main.py --mode train --checkpoint {interrupt_path}")

        # 训练结束
        self._cleanup()
        print(f"\n{'='*60}")
        print(f"训练结束 | 共 {episode + 1 - self.start_episode} episodes")
        print(f"最佳Avg分数: {best_avg_score:.1f}")
        print(f"模型保存在: {self.config.checkpoint_dir}/")
        print(f"{'='*60}")

        return dict(self.metrics)

    def load_checkpoint(self, path: str):
        """从检查点恢复训练"""
        result = load_checkpoint(self.agent, self.agent.optimizer, path, self.config.device)
        self.start_episode = result['episode']
        if 'metrics' in result:
            for key, values in result['metrics'].items():
                if key in self.metrics:
                    self.metrics[key] = values[:result['episode']]
        print(f"[恢复] 将从 episode {self.start_episode} 继续训练")

    def _cleanup(self):
        """清理资源"""
        if self.renderer:
            self.renderer.close()
            self.renderer = None
