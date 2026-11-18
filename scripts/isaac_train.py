#!/usr/bin/env python3
"""Isaac Lab 2.3.x / RSL-RL training adapter.

Launches the canonical train.py in an existing Isaac Lab checkout.
Requires ISAACLAB_ROOT and ISAAC_PYTHON, and optionally ISAACLAB_LOG_ROOT.
It finds a checkpoint by comparing pre/post file snapshots under a dedicated
experiment directory. Use a unique experiment_name to prevent collisions.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--params-json', type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--resume-checkpoint', default='', help='Optional champion checkpoint for fine-tuning')
    ap.add_argument('--task', required=True)
    ap.add_argument('--num-envs', type=int, default=4096)
    ap.add_argument('--max-iterations', type=int, default=10000)
    ap.add_argument('--experiment-name', default='rl_autolab')
    ap.add_argument('--map-json', type=Path, required=True)
    a=ap.parse_args()

    isaac_root=Path(os.environ.get('ISAACLAB_ROOT', '')).expanduser().resolve()
    if not os.environ.get('ISAACLAB_ROOT'):
        raise SystemExit('Please export ISAACLAB_ROOT=/path/to/IsaacLab')
    script=isaac_root/'scripts/reinforcement_learning/rsl_rl/train.py'
    if not script.is_file():
        raise SystemExit(f'Isaac Lab train.py not found: {script}')
    executable=os.environ.get('ISAAC_PYTHON', sys.executable)
    params=json.loads(a.params_json.read_text())
    mapping=json.loads(a.map_json.read_text())
    if set(mapping)!=set(params):
        raise SystemExit(f'Hydra map keys must equal params keys. map={set(mapping)}, params={set(params)}')
    overrides=[]
    for k, val in params.items():
        target=mapping[k]
        if not re.fullmatch(r'(?:env|agent)\.[a-zA-Z_0-9.]+', target):
            raise SystemExit(f'Unsafe or invalid Hydra override target: {target}')
        if type(val) not in (int,float):
            raise SystemExit(f'Non-numeric {k}')
        overrides.append(f'{target}={val}')

    # RSL-RL typically honors experiment_name and run_name from agent config.
    # Verify that custom task's Hydra config exposes these two fields.
    unique_run_name=f'{a.run_dir.parent.name}_{a.run_dir.name}_s{a.seed}'
    command=[executable, str(script), f'--task={a.task}', '--headless',
             f'--num_envs={a.num_envs}', f'--max_iterations={a.max_iterations}',
             f'--seed={a.seed}', f'agent.experiment_name={a.experiment_name}',
             f'agent.run_name={unique_run_name}', *overrides]

    log_root=Path(os.environ.get('ISAACLAB_LOG_ROOT',str(isaac_root/'logs/rsl_rl'))).expanduser().resolve()
    ckpt_root=log_root/a.experiment_name
    before={str(p.resolve()):p.stat().st_mtime_ns for p in ckpt_root.rglob('model_*.pt')} if ckpt_root.exists() else {}
    if a.resume_checkpoint:
        prev=Path(a.resume_checkpoint).resolve()
        if not prev.is_file():
            raise SystemExit(f'No resume checkpoint: {prev}')
        # v2.3.x RSL-RL runner resolves run folder and checkpoint using these fields.
        if ckpt_root.resolve() not in prev.parents:
            raise SystemExit(f'Resume checkpoint must be under experiment root: {ckpt_root}')
        command += ['agent.resume=true',f'agent.load_run={prev.parent.name}',
                    f'agent.load_checkpoint={prev.name}']
    print('Command:', ' '.join(command), flush=True)
    start=time.time()
    done=subprocess.run(command,cwd=isaac_root,check=False)
    if done.returncode:
        raise SystemExit(done.returncode)
    candidates=[]
    if ckpt_root.exists():
        for path in ckpt_root.rglob('model_*.pt'):
            if unique_run_name not in str(path):
                continue
            stamp=path.stat().st_mtime_ns
            if stamp>before.get(str(path.resolve()),0) and path.stat().st_mtime>=start-60:
                candidates.append(path)
    if not candidates:
        raise SystemExit(f'No checkpoint in {ckpt_root} for run {unique_run_name}. '
                         'Inspect your task agent.experiment_name/run_name and output paths.')
    checkpoint=max(candidates,key=lambda p:p.stat().st_mtime_ns).resolve()
    a.run_dir.mkdir(parents=True,exist_ok=True)
    (a.run_dir/'checkpoint_path.txt').write_text(str(checkpoint)+'\n')
    print('Checkpoint:',checkpoint,flush=True)

if __name__=='__main__':
    main()
