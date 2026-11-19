"""Tiny gridworld: reach GOAL from a random start. Shared by train.py and evaluate.py."""
SIZE = 15
GOAL = (SIZE - 1, SIZE - 1)
MAX_STEPS = 45
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def step(pos, action):
    dx, dy = ACTIONS[action]
    nxt = (min(SIZE - 1, max(0, pos[0] + dx)), min(SIZE - 1, max(0, pos[1] + dy)))
    return nxt, nxt == GOAL


def random_start(rng):
    while True:
        pos = (rng.randrange(SIZE), rng.randrange(SIZE))
        if pos != GOAL:
            return pos
