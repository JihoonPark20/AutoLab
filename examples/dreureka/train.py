#!/usr/bin/env python3
"""Q-learning in 'simulation': every episode samples slip ~ U(slip_min, slip_max) (domain randomization)."""
import argparse, importlib.util, json, pathlib, random
import env

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=pathlib.Path, required=True)
p.add_argument('--params-json', type=pathlib.Path, required=True)
p.add_argument('--code-file', type=pathlib.Path, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--episodes', type=int, default=400)
a = p.parse_args()
params = json.loads(a.params_json.read_text())
spec = importlib.util.spec_from_file_location('reward', a.code_file)
reward_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward_mod)

rng = random.Random(a.seed)
q = {}
alpha, gamma, eps = 0.5, 0.95, 0.1
with (a.run_dir / 'train_metrics.jsonl').open('w') as log:
    for ep in range(a.episodes):
        slip = rng.uniform(params['slip_min'], max(params['slip_min'], params['slip_max']))
        pos, totals, reached, fell = env.START, {}, False, False
        for t in range(env.MAX_STEPS):
            qs = q.setdefault(pos, [0.0] * 4)
            act = rng.randrange(4) if rng.random() < eps else max(range(4), key=lambda i: (qs[i], rng.random()))
            nxt, reached, fell = env.step(pos, act, slip, rng)
            r, parts = reward_mod.compute_reward({'pos': pos, 'next_pos': nxt, 'goal': env.GOAL,
                                                  'reached': reached, 'fell': fell, 'step': t})
            done = reached or fell
            target = r if done else r + gamma * max(q.setdefault(nxt, [0.0] * 4))
            qs[act] += alpha * (target - qs[act])
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            pos = nxt
            if done:
                break
        log.write(json.dumps({'episode': ep, 'slip': round(slip, 3), 'success': float(reached), 'fell': float(fell),
                              **{f'r_{k}': round(v, 4) for k, v in totals.items()}}) + '\n')
ckpt = a.run_dir / 'q_table.json'
ckpt.write_text(json.dumps({f'{x},{y}': v for (x, y), v in q.items()}))
(a.run_dir / 'checkpoint_path.txt').write_text(str(ckpt.resolve()))
