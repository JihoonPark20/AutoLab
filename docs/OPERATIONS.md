# Running Universal AutoLab through a terminal, Herdr, or Orca

AutoLab is a Python process; Herdr/Orca are optional operator interfaces.

## Recommended 2-pane workflow

- Pane A: open your research repo, run `claude` interactively for reviewing failure logs and code changes. Any code edits need review in a separate copy/worktree.
- Pane B: open the AutoLab directory, run `python rl_autolab.py run --config configs/my_project.json --state-dir runs/exp01 --cycles 2` under the project's activated Python environment. Training and evaluation run without continuous Claude turns.

If `backend=manual`, Pane A reads `runs/exp01/proposal_request.md` and writes `runs/exp01/proposal.json` in AutoLab's directory. Pane B reruns the same manager command.

If `backend=claude`, the manager directly executes `claude -p` only after evaluation. It does **not** send a message into the already-open interactive Claude pane in Herdr or Orca. The interactive pane is a separate session for observing/debugging.

## Ubuntu process management

A `systemd --user` service is an optional way to run the manager independent of a terminal; see `scripts/setup_systemd.py` for a template. Create absolute paths and activate the correct project Python before enabling it. The process does not bypass Claude account usage limits.

## Mac process management

`caffeinate -dimsu python rl_autolab.py run ...` can prevent idle sleep while attached, but closing a Mac laptop lid may suspend the machine. Check power/thermal settings for unattended runs.

## Git safety

The generic manager never invokes `git add`, `git commit`, `git checkout`, or `git worktree` automatically. An optional manual helper, `scripts/code_lane.py`, **does create Git worktree metadata** if you choose to run it. Never invoke it if you require a strictly untouched original `.git` directory. Prefer a disposable clone for code-edit experiments.
