"""所有超参数配置"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ========== 游戏设置 ==========
    grid_size: int = 15                # 网格大小 (grid_size × grid_size)
    max_steps_without_food: int = 100  # 饥饿超时步数

    # ========== 奖励设计 ==========
    reward_food: float = 10.0          # 吃到食物
    reward_collision: float = -10.0    # 碰撞（边界/自身）
    reward_step: float = 0.0           # 每步存活奖励（负值可防循环）
    reward_shaping: bool = False       # 是否使用距离塑形奖励
    reward_shaping_scale: float = 0.5  # 塑形奖励缩放

    # ========== 状态编码 ==========
    input_channels: int = 4            # 状态张量通道数

    # ========== DQN 超参数 ==========
    lr: float = 1e-4                   # 学习率
    gamma: float = 0.99                # 折扣因子
    batch_size: int = 64               # 批量大小
    buffer_capacity: int = 100_000     # 经验回放容量

    # ========== ε-贪心策略 ==========
    eps_start: float = 1.0             # 初始探索率
    eps_end: float = 0.01              # 最小探索率
    eps_decay: float = 0.995           # 每episode乘法衰减因子

    # ========== 目标网络 ==========
    target_update_freq: int = 10       # 每N个episode硬更新一次
    use_double_dqn: bool = True        # 是否使用Double DQN

    # ========== 训练设置 ==========
    n_episodes: int = 5000             # 总训练episode数
    warmup_steps: int = 1000           # 经验回放填充步数
    grad_clip_norm: float = 1.0        # 梯度裁剪范数

    # ========== 设备 ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 检查点 ==========
    checkpoint_dir: str = "./checkpoints"
    checkpoint_freq: int = 500

    # ========== 渲染 ==========
    render: bool = False
    render_fps: int = 30
    cell_size: int = 30
