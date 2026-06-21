"""模型检查点保存与加载"""

import os
import torch
from typing import Dict, Optional


def save_checkpoint(agent, metrics: Dict, episode: int,
                    checkpoint_dir: str = './checkpoints') -> str:
    """保存训练检查点

    Args:
        agent: DQNAgent 实例
        metrics: 训练指标字典
        episode: 当前 episode 编号
        checkpoint_dir: 检查点目录

    Returns:
        保存的文件路径
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    path = os.path.join(checkpoint_dir, f'snake_dqn_ep{episode}.pth')
    torch.save({
        'episode': episode,
        'online_state_dict': agent.online_net.state_dict(),
        'target_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'metrics': metrics,
        'epsilon': agent.epsilon_scheduler.get(),
        'grid_size': agent.config.grid_size,
        'input_channels': agent.config.input_channels,
        'n_actions': 4,
    }, path)

    print(f"[检查点] 已保存到 {path}")
    return path


def load_checkpoint(agent, optimizer, path: str, device: str = 'cpu') -> Dict:
    """加载训练检查点

    Args:
        agent: DQNAgent 实例（用于加载 online/target 网络）
        optimizer: 优化器实例
        path: 检查点文件路径
        device: 目标设备

    Returns:
        dict: 包含已保存的 episode 和 metrics
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"检查点文件不存在: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    agent.online_net.load_state_dict(checkpoint['online_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_state_dict'])
    agent.target_net.eval()

    # 尝试加载优化器状态（如果优化器结构相同）
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except Exception:
        print("[警告] 优化器状态加载失败，使用新优化器")

    episode = checkpoint.get('episode', 0)
    metrics = checkpoint.get('metrics', {})

    print(f"[检查点] 已从 {path} 加载 (episode {episode})")
    return {'episode': episode, 'metrics': metrics}


def find_latest_checkpoint(checkpoint_dir: str = './checkpoints') -> Optional[str]:
    """查找最新的检查点文件

    Args:
        checkpoint_dir: 检查点目录

    Returns:
        最新检查点路径，若不存在则返回 None
    """
    if not os.path.exists(checkpoint_dir):
        return None

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not files:
        return None

    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))
    return os.path.join(checkpoint_dir, files[-1])
