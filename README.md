<p align="center"><b>English</b> · <a href="README.ko.md">한국어</a> · <a href="README.zh.md">中文</a></p>

# AutoLab

<p align="center"><img src="docs/assets/teaser.svg" alt="AutoLab: manager loops between your RL project and a bounded proposal agent" width="100%"></p>

An unattended experiment loop for reinforcement-learning research. Any simulator and any OS that runs Python. Training takes hours, but deciding what to try next takes seconds. AutoLab runs train → fixed-seed eval → accept/reject → propose. The AI is called **once per finished experiment**, never while training runs. A Claude Code subscription's 5-hour and weekly limits can then cover a loop that runs for days.

- **Simulator-agnostic.** Your project only has to provide two commands that write `checkpoint_path.txt` and `eval.json`.
- **Zero dependencies.** One stdlib Python file (3.10+), macOS and Linux.
- **Bounded agent.** Proposals may only change allowlisted numeric parameters, within declared ranges and step sizes. The evaluator, seeds and metrics are frozen for the whole run.
- **Crash-safe.** Durable state, a lock per run directory and detached workers. Restarting never launches a duplicate GPU job.

## Quickstart

```bash
git clone https://github.com/JihoonPark20/AutoLab.git && cd AutoLab
python3 -m unittest discover -s tests          # 15 tests, no RL deps
python3 rl_autolab.py run --config configs/external_project_demo.json \
  --state-dir runs/demo --cycles 3              # synthetic toy project
python3 rl_autolab.py status --state-dir runs/demo
```

The demo uses a synthetic project and the `rules` backend. It checks the wiring only and is **not evidence of RL performance**.

## Connect your project

1. **Train command.** Reads `{params_json}` and `{seed}`, trains, then writes one line to `{run_dir}/checkpoint_path.txt` holding the checkpoint path.
2. **Eval command.** Loads `{checkpoint}`, runs your fixed protocol on `{fixed_seeds}`, then writes `{run_dir}/eval.json`:
   ```json
   {"evaluation_seeds": [201, 202, 203], "metrics": {"success_rate": 0.83, "fall_rate": 0.09}}
   ```
3. **Config.** Copy `configs/external_project_demo.json` and set `project.root`, `project.python`, the `train`/`eval` command arrays, `parameters` (`value`/`min`/`max`/`max_delta`), `decision` (primary metric, guardrails, target) and `limits`.

Commands are argument arrays, never run through a shell. Available tokens: `{python} {run_dir} {params_json} {seed} {fixed_seeds} {checkpoint} {resume_checkpoint} {project_root} {autolab_root}`, plus any key in `project.variables`. A reference adapter is in `examples/toy_project/`. Isaac Lab template: `docs/ISAAC_LAB.md`.

```bash
python3 rl_autolab.py doctor --config configs/my_project.json   # preflight, no training
```

## Run unattended

| `backend` | Who proposes the next experiment |
|---|---|
| `manual` | You. The loop pauses with `proposal_request.md`; write `proposal.json` and rerun the same command. |
| `claude` | `claude -p` (one turn, plan mode, JSON only), capped by `limits.max_calls_24h` / `max_calls_7d`. |
| `rules` | A fixed heuristic, for demos only. |

```bash
# macOS: caffeinate keeps the machine awake. Linux: see scripts/setup_systemd.py
caffeinate -dimsu python3 rl_autolab.py run --config configs/my_project.json \
  --state-dir runs/exp01 --cycles 40 --wait-budget 1800
```

`--wait-budget N` sleeps and retries when the local call cap or a Claude usage limit is hit, instead of exiting. Other commands: `stop`, `resume`, `status`, `approve-review --note ...`. Running `run` again continues from saved state; `--cycles` is the total target.

The loop stops on `target_score`, the cycle cap, repeated failures, or `max_consecutive_rejections`. On repeated rejections it pauses with `NEEDS_REVIEW` for a human.

## Safety boundaries

- The agent never edits source code. Reward, observation, termination or dynamics code changes go through `scripts/code_lane.py` (an isolated Git worktree) plus human review.
- `ANTHROPIC_API_KEY` in the environment is refused unless `allow_api_billing: true`. Call caps count local invocations, not tokens, and do not bypass provider limits.
- The run directory must sit outside `project.root`. `project.git_guard` rejects runs that change the research repo, but it is not a sandbox; use a disposable clone or container for untrusted code.
- The config is snapshotted when a run starts. Changing it requires a new `--state-dir`.

## Known limits

- Acceptance is primary metric + guardrails + `min_improvement`. That is an effect-size threshold, not a significance test. Use several `train_seeds` and an independent test seed grid before claiming results.
- In `finetune_from_champion` mode, a gain may come from extra training alone. Run a compute-matched control for causal claims.
- Interventions happen between runs. Changes in the middle of a rollout need a callback inside your training loop.
- Native Windows is unsupported (`fcntl`, POSIX process groups).

## Related work

Similar loops, where an LLM proposes and training feedback decides: **Eureka** (LLM-written rewards), **DrEureka** (domain randomization for sim-to-real), **Eurekaverse** (environment curricula) and Karpathy's **autoresearch** (an agent edits a training script under a fixed metric). AutoLab is the operations side of the same idea. It is simulator-agnostic, budgeted, crash-safe, and limited to a bounded parameter space by default.

More: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/OPERATIONS.md`](docs/OPERATIONS.md) · [`docs/ISAAC_LAB.md`](docs/ISAAC_LAB.md)

License: [MIT](LICENSE)
