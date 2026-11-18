#!/usr/bin/env python3
"""Create a real Herdr workspace with manager, log viewer and optional reviewer.

Confirmed against Herdr's JSON CLI contract (workspace create/tab create/pane run).
Requires Herdr running and its CLI on PATH. No synthetic/imaginary agent API.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rl_autolab import project_root, read_json

def call(*args):
    done=subprocess.run(["herdr",*args],text=True,capture_output=True)
    if done.returncode:
        raise RuntimeError(f"herdr {args}: {done.stderr[-600:] or done.stdout[-600:]}")
    return json.loads(done.stdout)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/demo.json')
    p.add_argument('--state-dir',default='runs/demo')
    p.add_argument('--cycles',type=int,default=5)
    p.add_argument('--reviewer',action='store_true',help='also start interactive Claude Code for manual code review')
    a=p.parse_args()
    if not shutil.which('herdr'):
        raise SystemExit('Install Herdr: https://herdr.dev/ and start `herdr` once.')
    config=(ROOT/a.config).resolve()
    state_dir=(ROOT/a.state_dir).resolve()
    if not config.is_file():
        raise SystemExit(f'Missing config: {config}')
    repo=project_root(read_json(config),config)
    work=call('workspace','create','--cwd',str(repo),'--label','RL-AutoLab','--no-focus')
    data=work['result']
    workspace=data['workspace']['workspace_id']
    manager_pane=data['root_pane']['pane_id']
    command=[sys.executable,'-u',str(ROOT/'rl_autolab.py'),'run',
             '--config',str(config),'--state-dir',str(state_dir),'--cycles',str(a.cycles)]
    call('pane','run',manager_pane,' '.join(shlex.quote(x) for x in command))
    status_tab=call('tab','create','--workspace',workspace,'--label','Experiment-status',
                    '--cwd',str(ROOT),'--no-focus')
    status_pane=status_tab['result']['root_pane']['pane_id']
    status_command=f"while true; do {shlex.quote(sys.executable)} {shlex.quote(str(ROOT/'rl_autolab.py'))} status --state-dir {shlex.quote(str(state_dir))}; sleep 10; done"
    call('pane','run',status_pane,status_command)
    if a.reviewer:
        reviewer_tab=call('tab','create','--workspace',workspace,'--label','Claude-review',
                          '--cwd',str(repo),'--no-focus')
        pane=reviewer_tab['result']['root_pane']['pane_id']
        call('agent','start','rl_reviewer','--kind','claude','--pane',pane)
    print('Workspace:',workspace)
    print('Manager pane:',manager_pane)
    print('Status pane:',status_pane)
    print('Attach with: herdr')
    print('Manager is independent of an interactive Claude session.\n')

if __name__=='__main__':
    main()
