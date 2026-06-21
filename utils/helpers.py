"""方向向量与坐标工具"""

# 动作映射: 0=上, 1=右, 2=下, 3=左
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

DIRECTION_VECTORS = {
    ACTION_UP: (-1, 0),
    ACTION_RIGHT: (0, 1),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
}

# 反向动作映射
OPPOSITE_ACTION = {
    ACTION_UP: ACTION_DOWN,
    ACTION_RIGHT: ACTION_LEFT,
    ACTION_DOWN: ACTION_UP,
    ACTION_LEFT: ACTION_RIGHT,
}

ACTION_NAMES = {0: "上", 1: "右", 2: "下", 3: "左"}


def is_opposite(action: int, current_action: int) -> bool:
    """判断 action 是否为 current_action 的反方向"""
    return action == OPPOSITE_ACTION.get(current_action, -1)


def manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    """计算曼哈顿距离"""
    return abs(r1 - r2) + abs(c1 - c2)
