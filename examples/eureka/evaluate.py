#!/usr/bin/env python3
"""Fixed fitness: greedy success from seeded starts. Never uses the reward file."""
import argparse, json, pathlib, random
import env

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=pathlib.Path, required=True)
p.add_argument('--checkpoint', type=pathlib.Path, required=True)
p.add_argument('--fixed-seeds', required=True)
a = p.parse_args()
q = {tuple(map(int, k.split(','))): v for k, v in json.loads(a.checkpoint.read_text()).items()}
seeds = [int(s) for s in a.fixed_seeds.split(',')]
per_seed, steps = {}, []
for seed in seeds:
    rng, wins = random.Random(seed), 0
    for _ in range(20):
        pos = env.random_start(rng)
        for t in range(env.MAX_STEPS):
            qs = q.get(pos)
            if not qs:
                break  # unvisited state: policy undefined
            pos, reached = env.step(pos, max(range(4), key=qs.__getitem__))
            if reached:
                wins += 1
                steps.append(t + 1)
                break
    per_seed[str(seed)] = wins / 20
metrics = {'success_rate': round(sum(per_seed.values()) / len(seeds), 4),
           'mean_steps_to_goal': round(sum(steps) / len(steps), 2) if steps else float(env.MAX_STEPS)}
(a.run_dir / 'eval.json').write_text(json.dumps({'evaluation_seeds': seeds, 'metrics': metrics,
                                                  'diagnostics': {'success_by_eval_seed': per_seed}}, indent=2))
print(metrics)
