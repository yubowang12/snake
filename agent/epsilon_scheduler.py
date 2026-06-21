"""ε-贪心探索率调度器"""


class EpsilonScheduler:
    """管理 ε 值的衰减

    支持两种衰减模式：
    - exponential: ε = max(ε_end, ε * decay_factor)  每步乘以衰减因子
    - linear: ε 线性从 start 降到 end，在 n_steps 内完成
    """

    def __init__(self, start: float = 1.0, end: float = 0.01,
                 decay: float = 0.995, mode: str = 'exponential'):
        """
        Args:
            start: 初始探索率
            end: 最小探索率
            decay: 衰减因子（仅 exponential 模式）
            mode: 'exponential' 或 'linear'
        """
        self.epsilon = start
        self.start = start
        self.end = end
        self.decay = decay
        self.mode = mode

        if mode == 'linear':
            # 线性模式需要外部设置 total_steps
            self.total_steps = 5000
            self.current_step = 0

    def set_linear_schedule(self, total_steps: int):
        """设置线性衰减的总步数"""
        self.total_steps = total_steps
        self.current_step = 0
        self.mode = 'linear'

    def step(self):
        """更新 ε 值（每个 episode 调用一次）"""
        if self.mode == 'exponential':
            self.epsilon = max(self.end, self.epsilon * self.decay)
        elif self.mode == 'linear':
            self.current_step += 1
            progress = min(1.0, self.current_step / self.total_steps)
            self.epsilon = self.start + (self.end - self.start) * progress
            self.epsilon = max(self.end, self.epsilon)

    def get(self) -> float:
        """获取当前 ε 值"""
        return self.epsilon

    def __repr__(self) -> str:
        return f"EpsilonScheduler(ε={self.epsilon:.4f}, mode={self.mode})"
