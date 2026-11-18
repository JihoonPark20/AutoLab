import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_autolab import generate_next_params, init_state, process_alive, read_json, validate_config


def fake_claude(bindir: Path, stdout: dict) -> None:
    exe = bindir / 'claude'
    exe.write_text(f'#!/bin/sh\ncat <<"EOF"\n{json.dumps(stdout)}\nEOF\n')
    exe.chmod(0o755)


class TestUnattended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = read_json(ROOT/'configs/demo.json')

    def test_exited_child_is_not_alive(self):
        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        time.sleep(0.5)  # exits, stays a zombie until reaped
        self.assertFalse(process_alive(p.pid))
        p.wait()

    def test_finetune_requires_train_seeds(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['training_mode'] = 'finetune_from_champion'
        with self.assertRaisesRegex(ValueError, 'train_seeds'):
            validate_config(cfg)

    def test_usage_limit_with_exit_zero_pauses_budget(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['backend'] = 'claude'
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fake_claude(td, {'type': 'result', 'is_error': True, 'result': 'Claude AI usage limit reached'})
            env = {'PATH': f"{td}:{os.environ['PATH']}"}
            st = init_state(cfg)
            with unittest.mock.patch.dict(os.environ, env), unittest.mock.patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}):
                self.assertFalse(generate_next_params(st, cfg, td, td/'state.json'))
            self.assertEqual(st['status'], 'PAUSED_BUDGET')
            self.assertGreater(st['next_ai_try_at'], time.time())

    def test_wait_budget_keeps_running_until_stop(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['backend'] = 'claude'
        cfg['limits']['max_calls_24h'] = 0  # budget exhausted from the start
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td/'cfg.json').write_text(json.dumps(cfg))
            state = td/'state'
            p = subprocess.Popen([sys.executable, str(ROOT/'rl_autolab.py'), 'run', '--config', str(td/'cfg.json'),
                                  '--state-dir', str(state), '--cycles', '3', '--wait-budget', '0.2'],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            deadline = time.time() + 30
            while not (state/'proposal_request.md').exists() and time.time() < deadline:
                time.sleep(0.2)
            time.sleep(0.5)
            self.assertIsNone(p.poll(), 'manager exited instead of waiting')
            (state/'STOP').write_text('x')
            out, _ = p.communicate(timeout=10)
            self.assertEqual(p.returncode, 2, out)
            self.assertIn('retrying', out)



if __name__ == '__main__':
    unittest.main()
