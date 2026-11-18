#!/usr/bin/env python3
"""Creates a Git worktree for supervised changes to a research environment.

No automatic merging. Tests and human review are the deliberate safety boundary.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def git(repo,*argv):
    p=subprocess.run(['git','-C',str(repo),*argv],text=True,capture_output=True)
    if p.returncode: raise RuntimeError(p.stderr or p.stdout)
    return p.stdout.strip()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True,help='YOUR Isaac Lab research code Git repository')
    p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--branch',default='autolab/mdp-improvement-001')
    p.add_argument('--worktree',type=Path,required=True)
    p.add_argument('--task',required=True,help='Hypothesis / intended MDP correction')
    a=p.parse_args()
    repo=a.repo.resolve()
    state=a.state_dir.resolve()
    worktree=a.worktree.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_./-]+',a.branch): raise ValueError('Unsafe branch')
    if git(repo,'status','--porcelain'):
        raise RuntimeError('Commit or stash research repo changes before creating worktree')
    if worktree.exists(): raise RuntimeError(f'Worktree path exists: {worktree}')
    data=json.loads((state/'state.json').read_text())
    if not data['history']: raise RuntimeError('Run at least one experiment before drafting code changes')
    report={'champion_run':data['champion_run'],'champion_params':data['champion_params'],
            'history':data['history'][-6:], 'evaluation_contract':'Do not alter held-out seeds or evaluation metric definitions'}
    git(repo,'worktree','add','-b',a.branch,str(worktree),'HEAD')
    text=(f'# Supervised RL Environment Improvement\n\n'
          f'## Hypothesis\n{a.task}\n\n'
          '## Required work\n'
          '- Inspect actual failure trajectories/logs and identify falsifiable MDP issue.\n'
          '- Edit reward, observation, termination, initial-state distribution, or dynamics ONLY as justified.\n'
          '- Add deterministic tests; run smoke tests before full GPU training.\n'
          '- Keep fixed-seed evaluation logic and metric names unchanged.\n'
          '- Do not merge into main or launch a large GPU run automatically.\n'
          '- Summarize diff, assumptions, and the experiment to test the hypothesis.\n\n'
          '## Experiment evidence (JSON)\n```json\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n```\n')
    (worktree/'AUTOLAB_CODE_TASK.md').write_text(text,encoding='utf8')
    print('Worktree:', worktree)
    print('Task:',worktree/'AUTOLAB_CODE_TASK.md')
    print('Run Claude Code in this isolated worktree; inspect git diff and run tests before merging.')

if __name__=='__main__':
    try: main()
    except (RuntimeError,ValueError,OSError) as e:
        print('ERROR:',e,file=sys.stderr); raise SystemExit(1)
