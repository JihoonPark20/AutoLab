# DrEureka: sim-to-real with LLM-chosen reward and domain randomization (cliff walk)

Training runs in a "simulation" whose slip probability is sampled per episode from `[slip_min, slip_max]`. Fitness is measured on held-out "real" dynamics (`real_slip = 0.2`) that the agent never sets. The agent controls both halves of DrEureka: the reward (the `code` slot, starting from `reward_v0.py`) and the randomization range (bounded `parameters`). The guardrail `sim_success_rate ≥ 0.9` stops it from trading away the task itself.

```bash
python3 rl_autolab.py run --config configs/dreureka.json --state-dir runs/dreureka   # no randomization: sim 1.00, real ≈ 0.52
cp examples/dreureka/proposal_dr.json runs/dreureka/proposal.json
python3 rl_autolab.py run --config configs/dreureka.json --state-dir runs/dreureka   # slip_max=0.2: real ≈ 0.91, TARGET_REACHED
```

`diagnostics` (per-seed real success and falls) and `train_curve` (`r_fall`, `slip`) give the planner the sim-to-real evidence. The paper's reward-aware physics prior (probing the policy across physics values before choosing ranges) is not reproduced. Here each range is judged by a full train-and-evaluate cycle.

Paper: *DrEureka: Language Model Guided Sim-To-Real Transfer* (Ma et al., RSS 2024, [arXiv:2406.01967](https://arxiv.org/abs/2406.01967)). Project page: https://eureka-research.github.io/dr-eureka/

This is a toy re-implementation of the method's loop for AutoLab, not a port of the official code.
