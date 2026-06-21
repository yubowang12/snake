"""DQN 卷积神经网络 —— 逼近 Q(s, a) 行为值函数"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """轻量级 CNN 用于逼近 Q 值函数

    输入: (batch, input_channels, H, W)  默认 (B, 4, 20, 20)
    输出: (batch, n_actions)             默认 (B, 4)

    架构: 3层卷积 (stride-2 降采样) → 自适应池化 → 2层全连接
    参数量约 28 万（比原版 1317 万减少 98%）。
    """

    def __init__(self, input_channels: int = 4, grid_size: int = 20,
                 n_actions: int = 4, dropout: float = 0.2):
        super().__init__()

        self.input_channels = input_channels
        self.grid_size = grid_size
        self.n_actions = n_actions

        # 卷积特征提取器 — stride=2 在第2层将 20→10
        self.conv = nn.Sequential(
            # Block 1: C→16, 保持 20×20
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2: 16→32, stride=2 降采样到 10×10
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: 32→64, 保持 10×10
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 自适应池化到固定小尺寸，使 FC 层与 grid_size 解耦
        self.pool = nn.AdaptiveAvgPool2d(5)  # → (64, 5, 5) = 1600

        conv_out_size = 64 * 5 * 5  # 1600

        # 全连接 Q 值头
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_actions),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming 权重初始化，适配 ReLU 激活"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 状态张量 (B, input_channels, H, W)

        Returns:
            Q值 (B, n_actions)
        """
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    def get_conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取池化前的卷积特征图（用于可视化分析）

        Args:
            x: 状态张量 (1, input_channels, H, W)

        Returns:
            卷积特征 (1, 64, H', W')
        """
        return self.conv(x)
