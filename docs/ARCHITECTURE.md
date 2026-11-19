# Universal AutoLab: architecture & integration contract

## Layers

```
               interactive Claude reviewer / model code developer
                                      |
                    separate external research project checkout
                                      |
     AutoLab manager (stdlib Python; independent of simulator/platform)
        + durable state, subprocess scheduling, fixed eval, scoring
        + bounded Claude Code JSON proposals, testable limits/permissions
                                      |
              subprocess train adapter + eval adapter
                   ├── MuJoCo / Stable-Baselines3 / RSL-RL
                   ├── Isaac Lab (requires task-specific evaluator)
                   └── any project implementing output contracts
```

## Project adapter API (CLI protocol)

All commands run with `cwd=project.root`, while `PYTHONPATH` includes the AutoLab installation root and project root. `PYTHONDONTWRITEBYTECODE=1` is set. The train adapter receives JSON parameters and fixed training seed; it writes a checkpoint path pointer file into an external run directory. The eval adapter receives a checkpoint and fixed eval seeds; it writes finite numeric metrics to a generic `eval.json`. No AutoLab core imports MuJoCo, torch, Isaac Lab or Claude SDK.

The adapter may use a project-specific process environment selected via `project.python` or by activating a virtualenv. Device settings (CPU/MPS/CUDA) belong in project variables or train/eval arguments, not in the core manager.

Minimal JSON config shape (omitting limits/targets here):

```json
{
  "project": {"root":"../Other_RL_Project", "python":"auto", "enforce_external_runs":true},
  "parameters": {"reward_scale":{"value":1.0,"min":0.5,"max":2.0,"max_delta":0.1}},
  "train": {"command":["{python}","train.py","--params-json","{params_json}","--run-dir","{run_dir}","--seed","{seed}"],"timeout_seconds":3600},
  "eval": {"command":["{python}","evaluate.py","--checkpoint","{checkpoint}","--run-dir","{run_dir}","--fixed-seeds","{fixed_seeds}"],"timeout_seconds":1200,"fixed_seeds":[42,43]},
  "decision": {"primary_metric":"success_rate","maximize":true,"min_improvement":0.01,"guardrails":{"fall_rate":{"max":0.1}}},
  "limits": {"max_calls_24h":3,"max_calls_7d":10}
}
```

Use full `configs/external_project_demo.json` as a real runnable config. One research repo = one config (and optionally an adapter). On Ubuntu, choose the appropriate conda interpreter. On macOS, activate the project's virtualenv. Simulators remain outside the manager.

## State machine and failure behavior

```
READY -> spawn WORKER -> TRAIN (train seeds) -> EVAL (fixed eval seeds)
                                      -> frozen worker_result.json
                              -> ACCEPT/REJECT -> PLAN (manual/Claude)
                              -> READY -> ...
```

Separate worker processes avoid keeping an interactive Claude session alive during hours-long RL. The worker process manages subprocess timeouts; the manager checks worker PID and result marker to avoid accidentally spawning duplicate long experiments on restart. `manager.lock` prevents two manager processes from changing the same run state simultaneously. STOP is applied between experiments. If a worker dies without a result marker, the run is marked failed.

When a proposal is generated, Python only permits allowlisted numeric keys, range bounds and maximum step size. Claude does not get permission to change its own safety/selection tests. The manager pauses on repeated failures, budget refusals or missing proposals.

## Research validity limits

The generic manager's acceptance rule is intentionally simple (primary scalar metric + guardrails). It is an exploratory experiment selector, **not a scientific proof** that an MDP change improved policy learning. Matched budget and training seeds are needed to separate the effect of a change from additional training. Fixed validation seeds are necessary for consistency but can themselves be overfit; independent test grids and multiple seeds remain essential.

With `training_mode=finetune_from_champion`, the source algorithm's `warm_start` or `resume` semantics are adapter-specific. When the environment distribution changes, prefer actor-only warm start over a full `--resume` that restores old MDP state; each adapter must declare its semantics.

## Git and file safety

By default AutoLab does not create Git branches, change tracked files or write ignore rules. It enforces that `state_dir` must be **outside** `project.root`, unless explicitly disabled. `project.git_guard=true` captures `git status --porcelain` before/after training and rejects a run if the status differs. This does not prevent writes, and `git status` may itself update index stat metadata. For *strict* original-repo protection, run adapters on a separate disposable copy or container and keep the original repo read-only.

`code_lane.py` is an optional legacy helper: its `git worktree add` command changes Git worktree metadata, so it must **not** be invoked when the original repository must remain untouched.

## OS support

Core manager supports POSIX macOS and Linux (Python 3.10+). macOS MPS, MuJoCo and Ubuntu CUDA are requirements of the connected project only. Native Windows would require replacing `fcntl` and POSIX process signals with cross-platform facilities.
