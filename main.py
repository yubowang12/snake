"""
贪吃蛇深度强化学习 (DQN) —— 主入口

用法:
  python main.py --mode train              # 训练模型
  python main.py --mode train --render     # 训练并可视化
  python main.py --mode eval --checkpoint checkpoints/best.pth   # 评估模型
  python main.py --mode demo --checkpoint checkpoints/best.pth   # 演示模式
  python main.py --mode analyze --checkpoint checkpoints/best.pth # 生成分析报告
"""

import sys
import os
import argparse

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from game.engine import SnakeEngine
from model.state_encoder import StateEncoder
from agent.dqn_agent import DQNAgent
from training.trainer import Trainer
from training.checkpoint import load_checkpoint, find_latest_checkpoint
from analysis.visualizer import Visualizer
from analysis.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(
        description='贪吃蛇 DQN 深度强化学习',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode train
  python main.py --mode train --render --episodes 2000
  python main.py --mode eval --checkpoint checkpoints/best.pth
  python main.py --mode demo --checkpoint checkpoints/best.pth
  python main.py --mode analyze --checkpoint checkpoints/best.pth
        """
    )
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'eval', 'demo', 'analyze'],
        help='运行模式: train(训练), eval(评估), demo(演示), analyze(分析)'
    )
    parser.add_argument(
        '--render', action='store_true',
        help='启用渲染（训练/评估模式下可视化游戏画面）'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='模型检查点路径（用于评估/演示/分析/续训）'
    )
    parser.add_argument(
        '--episodes', type=int, default=None,
        help='训练 episode 数（覆盖 config 中的默认值）'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='渲染帧率（默认30）'
    )

    args = parser.parse_args()

    config = Config()

    if args.render:
        config.render = True
    if args.episodes:
        config.n_episodes = args.episodes
    if args.fps:
        config.render_fps = args.fps

    print("=" * 60)
    print("  贪吃蛇 DQN 深度强化学习")
    print("=" * 60)
    print(f"  设备: {config.device}")
    print(f"  模式: {args.mode}")
    print("=" * 60)

    # ======== 训练模式 ========
    if args.mode == 'train':
        trainer = Trainer(config)
        if args.checkpoint:
            print(f"\n[续训] 从检查点恢复: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        metrics = trainer.train()

        # 训练结束后自动生成分析图
        print("\n[分析] 生成训练曲线...")
        evaluator = Evaluator(config)
        eval_stats = evaluator.evaluate(n_episodes=50, render=False)

        # 随机基线
        random_stats = Evaluator.evaluate_random(n_episodes=50, grid_size=config.grid_size)

        visualizer = Visualizer(save_dir='./plots')
        visualizer.plot_training_curves(metrics, save_path='./plots/training_curves.png')
        visualizer.plot_evaluation_results(
            eval_stats, random_stats, save_path='./plots/evaluation.png'
        )
        print(f"\n[完成] 训练曲线已保存到 ./plots/")

    # ======== 评估模式 ========
    elif args.mode == 'eval':
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
            if checkpoint_path is None:
                print("[错误] 未找到检查点，请先训练或通过 --checkpoint 指定路径")
                return

        evaluator = Evaluator(config)
        evaluator.load_model(checkpoint_path)

        if args.render:
            config.render = True

        eval_stats = evaluator.evaluate(n_episodes=100, render=args.render)

        # 对比随机基线
        print("\n[基线对比]")
        random_stats = Evaluator.evaluate_random(n_episodes=100, grid_size=config.grid_size)

        improvement = eval_stats['mean_score'] - random_stats['mean_score']
        print(f"\n  DQN vs 随机策略: +{improvement:.2f} 平均分数提升")

    # ======== 演示模式 ========
    elif args.mode == 'demo':
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
            if checkpoint_path is None:
                print("[错误] 未找到检查点，请先训练或通过 --checkpoint 指定路径")
                return

        config.render = True
        config.render_fps = 10  # 演示模式降低帧率
        evaluator = Evaluator(config)
        evaluator.demo(checkpoint_path)

    # ======== 分析模式 ========
    elif args.mode == 'analyze':
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
            if checkpoint_path is None:
                print("[错误] 未找到检查点，请先训练或通过 --checkpoint 指定路径")
                return

        evaluator = Evaluator(config)
        evaluator.load_model(checkpoint_path)

        # 运行评估
        eval_stats = evaluator.evaluate(n_episodes=100)
        random_stats = Evaluator.evaluate_random(n_episodes=100, grid_size=config.grid_size)

        # 加载历史训练指标
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        metrics = checkpoint.get('metrics', {})

        # 生成完整报告
        visualizer = Visualizer(save_dir='./plots')
        visualizer.generate_full_report(metrics, eval_stats)

        # 网络架构分析
        print("\n" + "=" * 60)
        print("  三方面分析报告")
        print("=" * 60)

        # A. 网络构建分析
        print("\n[A. 网络构建分析]")
        print(f"  CNN 架构: 3层卷积 (4→32→64→64) + 2层全连接 (25600→512→4)")
        print(f"  参数量: {sum(p.numel() for p in evaluator.agent.online_net.parameters()):,}")
        print(f"  正则化: BatchNorm3层 + Dropout(0.2)")
        print(f"  权重初始化: Kaiming Normal (适配ReLU)")
        print(f"  关键设计: 无池化层保持20×20完整空间分辨率")

        # B. 训练方法分析
        print("\n[B. 训练方法分析]")
        print(f"  算法: {'Double' if config.use_double_dqn else 'Vanilla'} DQN")
        print(f"  损失函数: SmoothL1Loss (Huber) — 比MSE更稳定")
        print(f"  目标网络更新: 每{config.target_update_freq}episodes硬更新")
        print(f"  ε策略: {config.eps_start}→{config.eps_end} 指数衰减 (×{config.eps_decay}/ep)")
        print(f"  经验回放容量: {config.buffer_capacity:,}")
        print(f"  批量大小: {config.batch_size}")
        print(f"  折扣因子 γ: {config.gamma}")
        print(f"  梯度裁剪: max_norm={config.grad_clip_norm}")

        # C. 性能表现分析
        print("\n[C. 性能表现分析]")
        print(f"  DQN 平均分数: {eval_stats['mean_score']:.2f}")
        print(f"  DQN 最大分数: {eval_stats['max_score']}")
        print(f"  DQN 平均步数: {eval_stats['mean_steps']:.1f}")
        print(f"  随机策略平均分数: {random_stats['mean_score']:.2f}")
        print(f"  随机策略最大分数: {random_stats['max_score']}")
        print(f"  相对提升: {(eval_stats['mean_score'] - random_stats['mean_score']):.2f} 分")
        print(f"  分数标准差: {eval_stats['std_score']:.2f}")

        if 'episode_reward' in metrics and len(metrics['episode_reward']) >= 100:
            import numpy as np
            rewards = np.array(metrics['episode_reward'])
            print(f"  训练收敛后平均奖励 (最后100ep): {rewards[-100:].mean():.2f}")
            if 'score' in metrics:
                scores = np.array(metrics['score'])
                print(f"  训练收敛后平均分数 (最后100ep): {scores[-100:].mean():.2f}")

        print(f"\n[分析] 图表已保存到 ./plots/")
        print("=" * 60)


if __name__ == '__main__':
    main()
