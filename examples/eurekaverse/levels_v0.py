def generate_level(rng, difficulty):
    """rng: random.Random; difficulty: 0.0 -> 1.0 over training. Returns {walls, start, goal}."""
    size = 10
    start = [rng.randrange(size), rng.randrange(size)]
    goal = [rng.randrange(size), rng.randrange(size)]
    while goal == start:
        goal = [rng.randrange(size), rng.randrange(size)]
    return {"walls": [], "start": start, "goal": goal}
