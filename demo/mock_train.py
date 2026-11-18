#!/usr/bin/env python3
"""Fake RL trace / checkpoint for integration tests; no RL is actually trained."""
import argparse, json, math, pathlib, random, time
p=argparse.ArgumentParser()
p.add_argument('--run-dir',type=pathlib.Path,required=True)
p.add_argument('--params-json',type=pathlib.Path,required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--resume-checkpoint',default='')
p.add_argument('--delay',type=float,default=0.001)
a=p.parse_args()
params=json.loads(a.params_json.read_text())
rng=random.Random(a.seed)
target=0.38+0.23*params['curriculum_difficulty']+0.18*params['reward_tracking']
out=a.run_dir/'train_metrics.jsonl'
with out.open('a') as f:
    for step in range(16):
        score=min(1,target*(1-math.exp(-step/5))+rng.uniform(-.012,.012))
        f.write(json.dumps({'iteration':step,'train_success':round(max(0,score),4)})+'\n')
        f.flush()
        # Model checkpoint written from step 12 onward, allowing plateau stopping.
        if step == 12:
            ckpt=a.run_dir/'checkpoint.mock'
            ckpt.write_text(json.dumps({'seed':a.seed,'params':params,'train_last':score, 'resumed_from':a.resume_checkpoint}))
            (a.run_dir/'checkpoint_path.txt').write_text(str(ckpt.resolve()))
        time.sleep(a.delay)
print('Mock training complete (no actual RL trained).')
