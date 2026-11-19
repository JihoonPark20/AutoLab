"""Obstacle grid. The agent only sees a local observation, so a policy can transfer across levels."""
SIZE = 10
MAX_STEPS = 50
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def blocked(level, cell):
    x, y = cell
    return not (0 <= x < SIZE and 0 <= y < SIZE) or cell in level["walls"]


def observe(level, pos):
    gx, gy = level["goal"]
    sign = lambda v: (v > 0) - (v < 0)
    return (sign(gx - pos[0]), sign(gy - pos[1]),
            *(int(blocked(level, (pos[0] + dx, pos[1] + dy))) for dx, dy in ACTIONS))


def step(level, pos, action):
    dx, dy = ACTIONS[action]
    nxt = (pos[0] + dx, pos[1] + dy)
    if blocked(level, nxt):
        nxt = pos
    return nxt, nxt == level["goal"]


def normalize(raw):
    """Validate a generated level: in-bounds cells, start/goal free and distinct."""
    walls = {tuple(map(int, c)) for c in raw.get("walls", [])}
    start, goal = tuple(map(int, raw["start"])), tuple(map(int, raw["goal"]))
    level = {"walls": {c for c in walls if 0 <= c[0] < SIZE and 0 <= c[1] < SIZE} - {start, goal},
             "start": start, "goal": goal}
    if start == goal or blocked({"walls": set()}, start) or blocked({"walls": set()}, goal):
        raise ValueError("generated level needs distinct in-bounds start and goal")
    return level
