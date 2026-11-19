"""Cliff walk: start (0,0), goal (11,0), cliff on (1..10, 0). With probability `slip` an action is replaced by a random one."""
WIDTH, HEIGHT = 12, 4
START, GOAL = (0, 0), (WIDTH - 1, 0)
CLIFF = {(x, 0) for x in range(1, WIDTH - 1)}
MAX_STEPS = 60
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def step(pos, action, slip, rng):
    if rng.random() < slip:
        action = rng.randrange(4)
    dx, dy = ACTIONS[action]
    nxt = (min(WIDTH - 1, max(0, pos[0] + dx)), min(HEIGHT - 1, max(0, pos[1] + dy)))
    return nxt, nxt == GOAL, nxt in CLIFF  # next, reached, fell
