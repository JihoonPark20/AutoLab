# Eurekaverse: an LLM writes the training environments (obstacle grid)

The agent sees only a local observation (direction to goal + its four neighbouring cells), so what it learns depends on which levels it trains on. The `code` slot here is not a reward. It is `generate_level(rng, difficulty)`, the training curriculum. Fitness is near-greedy (seeded ε = 0.1) success on fixed, solvable held-out levels with 20% obstacles that the generator never produces directly. The task reward and the learner are fixed.

```bash
python3 rl_autolab.py run --config configs/eurekaverse.json --state-dir runs/eurekaverse   # empty levels: held-out ≈ 0.34
cp examples/eurekaverse/proposal_curriculum.json runs/eurekaverse/proposal.json
python3 rl_autolab.py run --config configs/eurekaverse.json --state-dir runs/eurekaverse   # ramped obstacle curriculum: ≈ 0.57, TARGET_REACHED
```

`train_curve` reports `difficulty`, `walls`, `success` and `steps` per episode, so the planner can see whether a curriculum is too hard or too easy. The paper evolves several generators in parallel and selects by policy performance. Here one generator is tried per cycle.

Paper: *Eurekaverse: Environment Curriculum Generation via Large Language Models* (Liang et al., CoRL 2024, [arXiv:2411.01775](https://arxiv.org/abs/2411.01775)).
