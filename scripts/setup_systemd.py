#!/usr/bin/env python3
"""Write (but never start) a user systemd service + timer.
Run `systemctl --user ...` yourself after inspection.
"""
import argparse
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--service-name',default='rl-autolab')
    a=p.parse_args()
    if not a.service_name.replace('-','').isalnum():
        raise SystemExit('Invalid service name')
    config=a.config.resolve()
    dest=a.state_dir.resolve()
    if not config.is_file():
        raise SystemExit(f'No config: {config}')
    svc_dir=Path.home()/'.config/systemd/user'
    svc_dir.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(ROOT/'rl_autolab.py'),'run',
         '--config',str(config),'--state-dir',str(dest)]
    service=f'''[Unit]
Description=RL AutoLab Experiment Manager
After=default.target

[Service]
Type=oneshot
WorkingDirectory={ROOT}
ExecStart={' '.join(shlex.quote(x) for x in cmd)}
TimeoutStartSec=infinity
KillMode=process
SuccessExitStatus=2 3

[Install]
WantedBy=default.target
'''
    timer=f'''[Unit]
Description=Resume RL AutoLab regularly after budget / reviewer pause

[Timer]
OnBootSec=2min
OnUnitInactiveSec=30min
Unit={a.service_name}.service
Persistent=true

[Install]
WantedBy=timers.target
'''
    (svc_dir/f'{a.service_name}.service').write_text(service)
    (svc_dir/f'{a.service_name}.timer').write_text(timer)
    print('Wrote service/timer to',svc_dir)
    print('Review files then:')
    print('  systemctl --user daemon-reload')
    print(f'  systemctl --user enable --now {a.service_name}.timer')
    print(f'  journalctl --user -u {a.service_name}.service -f')
    print('For persistence after logout, Linux may require loginctl enable-linger (admin permission).')

if __name__=='__main__':main()
