#!/usr/bin/env python3
"""Fixed-seed synthetic evaluator to test score/guardrails; NOT an IsaacLab eval."""
import argparse,json,random,pathlib
p=argparse.ArgumentParser()
p.add_argument('--run-dir',type=pathlib.Path,required=True)
p.add_argument('--params-json',type=pathlib.Path,required=True)
p.add_argument('--checkpoint',type=pathlib.Path,required=True)
p.add_argument('--fixed-seeds',required=True)
a=p.parse_args()
params=json.loads(a.params_json.read_text())
ckpt=json.loads(a.checkpoint.read_text())
assert ckpt['params']==params
seeds=[int(x) for x in a.fixed_seeds.split(',')]
values=[]
for seed in seeds:
    rng=random.Random(seed)
    # Training on a gradually broader distribution improves synthetic held-out success.
    val=0.41+0.55*params['curriculum_difficulty']+0.17*(params['reward_tracking']-1)
    # Eventually extreme curriculum hurts; creates non-monotone optimization problem.
    val-=0.60*max(0,params['curriculum_difficulty']-0.70)**2
    values.append(min(1,max(0,val+rng.uniform(-.015,.015))))
score=sum(values)/len(values)
metrics={'success_rate':round(score,4),
         'fall_rate':round(max(0.0,0.24-0.14*params['curriculum_difficulty']),4),
         'energy':round(1.1+0.1*(params['reward_tracking']-1)**2,4),
         'worst_seed_success':round(min(values),4)}
(a.run_dir/'eval.json').write_text(json.dumps({'evaluation_seeds':seeds,'metrics':metrics},indent=2)+'\n')
print('Synthetic held-out evaluation:', metrics)
