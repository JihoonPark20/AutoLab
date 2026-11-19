#!/usr/bin/env python3
"""Fitness on held-out 'real' dynamics (slip = --real-slip), plus the sim (slip 0) score to expose the gap."""
import argparse, json, pathlib, random
import env

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=pathlib.Path, required=True)
p.add_argument('--checkpoint', type=pathlib.Path, required=True)
p.add_argument('--fixed-seeds', required=True)
p.add_argument('--real-slip', type=float, default=0.2)
a = p.parse_args()
q = {tuple(map(int, k.split(','))): v for k, v in json.loads(a.checkpoint.read_text()).items()}
seeds = [int(s) for s in a.fixed_seeds.split(',')]


def rollout_success(slip, seed, episodes=50):
    rng, wins, falls = random.Random(seed), 0, 0
    for _ in range(episodes):
        pos = env.START
        for _ in range(env.MAX_STEPS):
            qs = q.get(pos, [0.0] * 4)
            pos, reached, fell = env.step(pos, max(range(4), key=qs.__getitem__), slip, rng)
            if reached or fell:
                wins += reached
                falls += fell
                break
    return wins / episodes, falls / episodes


real = {str(s): rollout_success(a.real_slip, s) for s in seeds}
sim = [rollout_success(0.0, s)[0] for s in seeds]
metrics = {'success_rate': round(sum(v[0] for v in real.values()) / len(seeds), 4),
           'fall_rate': round(sum(v[1] for v in real.values()) / len(seeds), 4),
           'sim_success_rate': round(sum(sim) / len(seeds), 4)}
(a.run_dir / 'eval.json').write_text(json.dumps({'evaluation_seeds': seeds, 'metrics': metrics,
    'diagnostics': {'real_slip': a.real_slip, 'real_success_fall_by_seed': real}}, indent=2))
print(metrics)
