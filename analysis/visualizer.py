"""训练分析与可视化

从以下三个方面生成分析图表：

A. 网络构建分析
   - 不同CNN架构对比（卷积层数、通道数影响）
   - BatchNorm / Dropout 消融实验
   - 卷积层激活图可视化

B. 训练方法分析
   - Double DQN vs Vanilla DQN 对比
   - 目标网络更新频率对比
   - ε 衰减策略对比
   - 经验回放缓冲区大小影响

C. 性能表现分析
   - 训练曲线（奖励、分数、损失、ε衰减）
   - 评估基准统计
   - 与随机策略对比
   - TD 误差分布
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 依赖
import matplotlib.pyplot as plt
import torch
from typing import Dict, Optional, List


# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """训练过程可视化工具"""

    def __init__(self, save_dir: str = './plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # ================================================================
    # A. 网络构建分析
    # ================================================================

    @staticmethod
    def plot_architecture_comparison(
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """对比不同网络架构的训练效果

        Args:
            results: {arch_name: {'rewards': [...], 'scores': [...]}}
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for name, data in results.items():
            rewards = data['rewards']
            smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
            axes[0].plot(smoothed, label=name, alpha=0.8)
        axes[0].set_title('不同网络架构的奖励曲线对比')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('平滑奖励 (窗口=50)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for name, data in results.items():
            scores = data['scores']
            smoothed = np.convolve(scores, np.ones(50)/50, mode='valid')
            axes[1].plot(smoothed, label=name, alpha=0.8)
        axes[1].set_title('不同网络架构的分数曲线对比')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('平滑分数 (窗口=50)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_ablation_study(
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """消融实验对比图（BatchNorm/Dropout等）

        Args:
            results: {condition: {'final_avg_score': float, 'final_avg_reward': float}}
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        conditions = list(results.keys())
        x = np.arange(len(conditions))
        width = 0.35

        scores = [results[c]['final_avg_score'] for c in conditions]
        rewards = [results[c]['final_avg_reward'] for c in conditions]

        bars1 = ax.bar(x - width/2, scores, width, label='平均分数', color='#2ecc71')
        ax_twin = ax.twinx()
        bars2 = ax_twin.bar(x + width/2, rewards, width, label='平均奖励', color='#3498db')

        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel('平均分数', color='#2ecc71')
        ax_twin.set_ylabel('平均奖励', color='#3498db')
        ax.set_title('消融实验对比')

        # 添加数值标签
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

        fig.legend(loc='upper right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_conv_activations(
        model,
        state: torch.Tensor,
        layer_idx: int = -1,
        save_path: Optional[str] = None
    ):
        """可视化卷积层激活图

        Args:
            model: DQNNetwork 实例
            state: 单个状态 (1, C, H, W)
            layer_idx: 要可视化的卷积层索引 (0, 1, 2)
            save_path: 保存路径
        """
        model.eval()
        with torch.no_grad():
            # 逐层提取特征
            x = state
            target_conv = model.conv[layer_idx * 3]  # Conv2d 层

            # 通过前面的层
            for i, layer in enumerate(model.conv):
                x = layer(x)
                if i == layer_idx * 3:  # 达到目标 Conv2d 层
                    break
                if i > layer_idx * 3 + 2:  # 超过目标 block
                    break

        activations = x.squeeze(0).cpu().numpy()
        n_channels = activations.shape[0]

        # 显示前16个通道
        n_cols = 8
        n_rows = min(2, (n_channels + n_cols - 1) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else axes

        for i in range(n_rows * n_cols):
            if i < n_channels:
                axes[i].imshow(activations[i], cmap='viridis')
                axes[i].set_title(f'通道 {i}')
            else:
                axes[i].axis('off')
            axes[i].axis('off')

        plt.suptitle(f'卷积层 {layer_idx + 1} 激活图', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    # ================================================================
    # B. 训练方法分析
    # ================================================================

    @staticmethod
    def plot_training_curves(
        metrics: Dict,
        save_path: Optional[str] = None
    ):
        """4合1训练曲线图：
        1) Episode 奖励
        2) 分数（吃到的食物数量）
        3) 训练损失
        4) ε 衰减
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: Episode 奖励
        rewards = metrics['episode_reward']
        if len(rewards) > 50:
            smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
            axes[0, 0].plot(rewards, alpha=0.2, color='#3498db', linewidth=0.5)
            axes[0, 0].plot(
                range(49, len(rewards)), smoothed,
                color='#3498db', linewidth=1.5, label='平滑 (窗口=50)'
            )
        else:
            axes[0, 0].plot(rewards, color='#3498db')
        axes[0, 0].set_title('Episode 奖励')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 子图2: 分数
        scores = metrics['score']
        if len(scores) > 50:
            smoothed_scores = np.convolve(scores, np.ones(50)/50, mode='valid')
            axes[0, 1].plot(scores, alpha=0.2, color='#2ecc71', linewidth=0.5)
            axes[0, 1].plot(
                range(49, len(scores)), smoothed_scores,
                color='#2ecc71', linewidth=1.5, label='平滑 (窗口=50)'
            )
        else:
            axes[0, 1].plot(scores, color='#2ecc71')
        axes[0, 1].set_title('分数 (吃到的食物)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('分数')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 子图3: 训练损失（对数坐标）
        losses = metrics['avg_loss']
        axes[1, 0].plot(losses, color='#e74c3c', linewidth=0.8)
        axes[1, 0].set_title('平均 TD 损失')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # 子图4: ε 衰减
        epsilons = metrics['epsilon']
        axes[1, 1].plot(epsilons, color='#9b59b6', linewidth=1.5)
        axes[1, 1].set_title('ε 探索率衰减')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('ε')
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('DQN 训练过程分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_method_comparison(
        results: Dict[str, Dict],
        metric: str = 'rewards',
        save_path: Optional[str] = None
    ):
        """对比不同训练方法的训练曲线

        Args:
            results: {method_name: {metric: [...]}}
            metric: 'rewards' 或 'scores'
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
        for i, (name, data) in enumerate(results.items()):
            values = data[metric]
            if len(values) > 50:
                smoothed = np.convolve(values, np.ones(50)/50, mode='valid')
                ax.plot(
                    range(49, len(values)), smoothed,
                    color=colors[i % len(colors)], linewidth=1.5, label=name
                )
            else:
                ax.plot(values, color=colors[i % len(colors)], label=name)

        metric_name = '奖励' if metric == 'rewards' else '分数'
        ax.set_title(f'训练方法对比 — {metric_name}曲线')
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    # ================================================================
    # C. 性能表现分析
    # ================================================================

    @staticmethod
    def plot_evaluation_results(
        eval_stats: Dict,
        random_baseline: Optional[Dict] = None,
        save_path: Optional[str] = None
    ):
        """评估结果对比图

        Args:
            eval_stats: {'mean_score': float, 'max_score': int, 'std_score': float, ...}
            random_baseline: 随机策略的相同统计量
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 子图1: 平均/最大分数对比
        categories = ['平均分数', '最大分数']
        dqn_values = [eval_stats['mean_score'], eval_stats['max_score']]
        if random_baseline:
            random_values = [random_baseline['mean_score'], random_baseline['max_score']]
        else:
            random_values = [0, 0]  # 随机策略通常吃不到食物

        x = np.arange(len(categories))
        width = 0.3

        axes[0].bar(x - width/2, dqn_values, width, label='DQN 智能体', color='#2ecc71')
        axes[0].bar(x + width/2, random_values, width, label='随机策略', color='#e74c3c')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].set_ylabel('分数')
        axes[0].set_title('DQN vs 随机策略')
        axes[0].legend()

        # 添加数值标签
        for i, (d, r) in enumerate(zip(dqn_values, random_values)):
            axes[0].text(i - width/2, d + 0.1, f'{d:.1f}', ha='center', fontsize=10)
            axes[0].text(i + width/2, r + 0.1, f'{r:.1f}', ha='center', fontsize=10)

        # 子图2: 统计分布
        stats_labels = ['平均存活步数', '标准差']
        dqn_stats = [eval_stats.get('mean_steps', 0), eval_stats.get('std_score', 0)]
        x2 = np.arange(len(stats_labels))
        axes[1].bar(x2, dqn_stats, color='#3498db')
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(stats_labels)
        axes[1].set_title('评估统计量')
        for i, v in enumerate(dqn_stats):
            axes[1].text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_td_error_distribution(
        td_errors: List[float],
        save_path: Optional[str] = None
    ):
        """TD 误差分布直方图

        Args:
            td_errors: TD 误差列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 直方图
        axes[0].hist(td_errors, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
        axes[0].axvline(np.mean(td_errors), color='red', linestyle='--',
                        label=f'均值: {np.mean(td_errors):.4f}')
        axes[0].set_title('TD 误差分布')
        axes[0].set_xlabel('TD 误差')
        axes[0].set_ylabel('频次')
        axes[0].legend()

        # Q-Q 图检验正态性
        from scipy import stats as scipy_stats
        sorted_errors = np.sort(td_errors)
        theoretical = scipy_stats.norm.ppf(
            (np.arange(len(td_errors)) + 0.5) / len(td_errors)
        )
        axes[1].scatter(theoretical, sorted_errors, alpha=0.5, s=10)
        axes[1].plot(
            [theoretical.min(), theoretical.max()],
            [theoretical.min(), theoretical.max()],
            'r--', linewidth=1
        )
        axes[1].set_title('Q-Q 图 (vs 正态分布)')
        axes[1].set_xlabel('理论分位数')
        axes[1].set_ylabel('样本分位数')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    # ================================================================
    # 综合报告
    # ================================================================

    def generate_full_report(self, metrics: Dict, eval_stats: Dict,
                             save_dir: Optional[str] = None):
        """生成完整的分析报告（所有图表）"""
        if save_dir is None:
            save_dir = self.save_dir

        print("\n[分析] 生成训练曲线图...")
        self.plot_training_curves(
            metrics,
            save_path=os.path.join(save_dir, 'training_curves.png')
        )

        print("[分析] 生成评估结果图...")
        self.plot_evaluation_results(
            eval_stats,
            save_path=os.path.join(save_dir, 'evaluation.png')
        )

        print(f"[分析] 所有图表已保存到 {save_dir}/")
