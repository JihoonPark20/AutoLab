#!/usr/bin/env python3
"""Fitness: greedy success on fixed held-out obstacle levels (solvable, 20% walls). Never runs the generator."""
import argparse, collections, json, pathlib, random
import env

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=pathlib.Path, required=True)
p.add_argument('--checkpoint', type=pathlib.Path, required=True)
p.add_argument('--fixed-seeds', required=True)
a = p.parse_args()
DENSITY, EVAL_EPS = 0.2, 0.1  # fixed protocol: a small seeded exploration breaks greedy loops
q = {tuple(k): v for k, v in json.loads(a.checkpoint.read_text())}
seeds = [int(s) for s in a.fixed_seeds.split(',')]


def solvable(level):
    seen, todo = {level['start']}, collections.deque([level['start']])
    while todo:
        pos = todo.popleft()
        if pos == level['goal']:
            return True
        for act in range(4):
            nxt, _ = env.step(level, pos, act)
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return False


def held_out_level(rng):
    while True:
        cells = [(x, y) for x in range(env.SIZE) for y in range(env.SIZE)]
        start, goal = rng.sample(cells, 2)
        level = {'walls': {c for c in cells if rng.random() < DENSITY} - {start, goal}, 'start': start, 'goal': goal}
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) >= 6 and solvable(level):
            return level


per_seed = {}
for seed in seeds:
    rng, wins = random.Random(seed), 0
    for _ in range(30):
        level = held_out_level(rng)
        pos = level['start']
        for _ in range(env.MAX_STEPS):
            qs = q.get(env.observe(level, pos), [0.0] * 4)
            act = rng.randrange(4) if rng.random() < EVAL_EPS else max(range(4), key=qs.__getitem__)
            pos, reached = env.step(level, pos, act)
            if reached:
                wins += 1
                break
    per_seed[str(seed)] = round(wins / 30, 4)
metrics = {'success_rate': round(sum(per_seed.values()) / len(seeds), 4)}
(a.run_dir / 'eval.json').write_text(json.dumps({'evaluation_seeds': seeds, 'metrics': metrics,
                                                  'diagnostics': {'held_out_success_by_seed': per_seed}}, indent=2))
print(metrics)
