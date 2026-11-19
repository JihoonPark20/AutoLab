#!/usr/bin/env python3
"""Q-learning over local observations on levels produced by a generator that AutoLab supplies via {code_file}."""
import argparse, importlib.util, json, pathlib, random
import env

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=pathlib.Path, required=True)
p.add_argument('--params-json', type=pathlib.Path, required=True)
p.add_argument('--code-file', type=pathlib.Path, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--episodes', type=int, default=1500)
a = p.parse_args()
params = json.loads(a.params_json.read_text())
spec = importlib.util.spec_from_file_location('levels', a.code_file)
levels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(levels)

rng, level_rng = random.Random(a.seed), random.Random(a.seed + 1000)
q = {}
alpha, gamma, eps = params['learning_rate'], 0.5, 0.2
with (a.run_dir / 'train_metrics.jsonl').open('w') as log:
    for ep in range(a.episodes):
        difficulty = ep / max(1, a.episodes - 1)
        level = env.normalize(levels.generate_level(level_rng, difficulty))
        pos, reached = level['start'], False
        for t in range(env.MAX_STEPS):
            obs = env.observe(level, pos)
            qs = q.setdefault(obs, [0.0] * 4)
            act = rng.randrange(4) if rng.random() < eps else max(range(4), key=lambda i: (qs[i], rng.random()))
            nxt, reached = env.step(level, pos, act)
            gx, gy = level['goal']
            progress = (abs(gx - pos[0]) + abs(gy - pos[1])) - (abs(gx - nxt[0]) + abs(gy - nxt[1]))
            r = 1.0 if reached else 0.1 * progress - 0.02  # fixed task reward; only the levels are searched
            target = r if reached else r + gamma * max(q.setdefault(env.observe(level, nxt), [0.0] * 4))
            qs[act] += alpha * (target - qs[act])
            pos = nxt
            if reached:
                break
        log.write(json.dumps({'episode': ep, 'difficulty': round(difficulty, 3), 'walls': len(level['walls']),
                              'success': float(reached), 'steps': t + 1}) + '\n')
ckpt = a.run_dir / 'q_table.json'
ckpt.write_text(json.dumps([[list(k), v] for k, v in q.items()]))
(a.run_dir / 'checkpoint_path.txt').write_text(str(ckpt.resolve()))
