def compute_reward(obs):
    """obs: pos, next_pos, goal, size, reached, step. Returns (reward, components)."""
    success = 1.0 if obs["reached"] else 0.0
    return success, {"success": success}
