# Reward weight tuning (gridworld)

The most common reward-design loop in practice: the reward's structure is fixed (`goal + progress + step cost`) and only its weights move. Weights are ordinary AutoLab `parameters`, so every proposal is checked against bounds and `max_delta`, and only one weight may change per step (`max_changed_parameters: 1`). No code is generated.

```bash
python3 rl_autolab.py run --config configs/reward_weights.json --state-dir runs/weights   # w_progress=0: success ≈ 0.27
cp examples/reward_weights/proposal_1_progress.json runs/weights/proposal.json
python3 rl_autolab.py run --config configs/reward_weights.json --state-dir runs/weights   # w_progress=0.05: ≈ 0.96
cp examples/reward_weights/proposal_2_step.json runs/weights/proposal.json
python3 rl_autolab.py run --config configs/reward_weights.json --state-dir runs/weights   # w_step=-0.05: ≈ 0.99, TARGET_REACHED
```

Per-episode `r_goal`, `r_progress` and `r_step` sums reach the planner as `train_curve`. Set `"backend": "claude"` to let the agent choose the weights.
