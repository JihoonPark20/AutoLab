# Eureka: reward design by an LLM (gridworld)

A real (tiny) RL task: tabular Q-learning must reach a goal on a 15×15 grid from random starts in 120 training episodes. Training reads its reward from `{code_file}`, which AutoLab writes for each run. `evaluate.py` measures greedy success on fixed seeds and never uses the reward. That fixed fitness is what keeps reward search honest.

- `reward_sparse.py`: the starting reward (+1 at the goal). It learns slowly, at about 0.3 success.
- `proposal_shaped.json`: an example proposal carrying `code`, the format the `claude` backend returns.
- `train_metrics.jsonl`: per-episode sums of each reward component (`r_<name>`). This is the reward reflection the planner sees.

Paper: *Eureka: Human-Level Reward Design via Coding Large Language Models* (Ma et al., ICLR 2024, [arXiv:2310.12931](https://arxiv.org/abs/2310.12931)). Project page: https://eureka-research.github.io/

This is a toy re-implementation of the method's loop for AutoLab, not a port of the official code.

```bash
python3 rl_autolab.py run --config configs/eureka.json --state-dir runs/eureka   # sparse reward: success ≈ 0.27
cp examples/eureka/proposal_shaped.json runs/eureka/proposal.json
python3 rl_autolab.py run --config configs/eureka.json --state-dir runs/eureka   # shaped reward: ≈ 0.94, TARGET_REACHED
```

Set `"backend": "claude"` in `configs/eureka.json` to let the agent write the reward itself.
