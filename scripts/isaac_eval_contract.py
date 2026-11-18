#!/usr/bin/env python3
"""Adapter contract for a project-specific *held-out* Isaac Lab roll-out.

Do NOT use training TensorBoard reward as evaluation. Replace evaluate_policy()
with a call to your task's rollout/play-based evaluator. This deliberately fails
until a real task-specific rollout is supplied; it cannot fake metrics.

Your evaluator must return dict of scalar metrics: success_rate, fall_rate,
energy and any project-specific constraint metrics, evaluated using the fixed
seeds passed here. The caller validates reported seeds.
"""
import argparse
import importlib
import os
import json
from pathlib import Path


def evaluate_policy(checkpoint: Path, task: str, seeds: list[int]) -> dict[str, float]:
    plugin=os.environ.get('RL_EVAL_FUNCTION', '')
    if not plugin or ':' not in plugin:
        raise NotImplementedError(
            'Set RL_EVAL_FUNCTION=your_package.eval:evaluate_policy to connect held-out '
            'Isaac Lab rollouts (checkpoint, task, seeds) -> metrics dictionary. '
            'See docs/ISAAC_LAB.md. Never substitute a fabricated metric.'
        )
    module, function = plugin.split(':', 1)
    target=getattr(importlib.import_module(module), function)
    return target(checkpoint, task, seeds)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--task',required=True)
    p.add_argument('--fixed-seeds',required=True)
    a=p.parse_args()
    seeds=[int(x) for x in a.fixed_seeds.split(',')]
    metrics=evaluate_policy(a.checkpoint,a.task,seeds)
    for key in ('success_rate','fall_rate','energy'):
        if key not in metrics:
            raise ValueError(f'Missing metric {key}')
    (a.run_dir/'eval.json').write_text(json.dumps({'evaluation_seeds':seeds,'metrics':metrics},indent=2)+'\n')

if __name__=='__main__':
    main()
