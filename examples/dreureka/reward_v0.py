def compute_reward(obs):
    """obs: pos, next_pos, goal, reached, fell, step. Returns (reward, components)."""
    parts = {"goal": 1.0 if obs["reached"] else 0.0, "fall": -1.0 if obs["fell"] else 0.0, "step": -0.01}
    return sum(parts.values()), parts
