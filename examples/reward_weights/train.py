#!/usr/bin/env python3
"""Tabular Q-learning with a fixed reward structure whose weights come from AutoLab parameters."""
import argparse, json, pathlib, random
import env

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=pathlib.Path, required=True)
p.add_argument('--params-json', type=pathlib.Path, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--episodes', type=int, default=120)
a = p.parse_args()
w = json.loads(a.params_json.read_text())


def reward(pos, nxt, reached):
    gx, gy = env.GOAL
    progress = (abs(gx - pos[0]) + abs(gy - pos[1])) - (abs(gx - nxt[0]) + abs(gy - nxt[1]))
    parts = {'goal': w['w_goal'] * reached, 'progress': w['w_progress'] * progress, 'step': w['w_step']}
    return sum(parts.values()), parts


rng = random.Random(a.seed)
q = {}
alpha, gamma, eps = 0.5, 0.95, 0.2
with (a.run_dir / 'train_metrics.jsonl').open('w') as log:
    for ep in range(a.episodes):
        pos, totals, reached = env.random_start(rng), {}, False
        for t in range(env.MAX_STEPS):
            qs = q.setdefault(pos, [0.0] * 4)
            act = rng.randrange(4) if rng.random() < eps else max(range(4), key=lambda i: (qs[i], rng.random()))
            nxt, reached = env.step(pos, act)
            r, parts = reward(pos, nxt, reached)
            target = r if reached else r + gamma * max(q.setdefault(nxt, [0.0] * 4))
            qs[act] += alpha * (target - qs[act])
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            pos = nxt
            if reached:
                break
        log.write(json.dumps({'episode': ep, 'success': float(reached),
                              **{f'r_{k}': round(v, 4) for k, v in totals.items()}}) + '\n')
ckpt = a.run_dir / 'q_table.json'
ckpt.write_text(json.dumps({f'{x},{y}': v for (x, y), v in q.items()}))
(a.run_dir / 'checkpoint_path.txt').write_text(str(ckpt.resolve()))
