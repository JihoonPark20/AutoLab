import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_autolab import as_command, read_json, render_prompt


class TestFeedback(unittest.TestCase):
    def run_demo(self, cfg, td, cycles='2'):
        (td/'cfg.json').write_text(json.dumps(cfg))
        got = subprocess.run([sys.executable, str(ROOT/'rl_autolab.py'), 'run', '--config', str(td/'cfg.json'),
                              '--state-dir', str(td/'state'), '--cycles', cycles],
                             text=True, capture_output=True, timeout=35)
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        return read_json(td/'state'/'state.json')

    def test_diagnostics_and_train_curve_reach_prompt(self):
        cfg = read_json(ROOT/'configs/demo.json')
        with tempfile.TemporaryDirectory() as td:
            st = self.run_demo(cfg, Path(td))
            diag = st['history'][-1]['diagnostics'][0]
            self.assertIn('success_by_eval_seed', diag['diagnostics'])
            curve = diag['train_curve']['train_success']
            self.assertEqual(curve['n'], 16)
            self.assertGreater(curve['last'], curve['first'])
            prompt = render_prompt(st, cfg)
            self.assertIn('success_by_eval_seed', prompt)
            self.assertIn('train_curve', prompt)

    def test_notify_on_exit(self):
        cfg = copy.deepcopy(read_json(ROOT/'configs/demo.json'))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'note.txt'
            cfg['notify'] = {'command': [sys.executable, '-c', 'import sys; open(sys.argv[1], "w").write(sys.argv[2])',
                                         str(out), '{message}']}
            self.run_demo(cfg, Path(td), cycles='1')
            self.assertIn('MAX_CYCLES_REACHED', out.read_text())

    def test_token_values_are_not_rescanned(self):
        self.assertEqual(as_command(['{message}'], {'message': 'reason {weird}'}), ['reason {weird}'])
        with self.assertRaises(ValueError):
            as_command(['{missing}'], {})


if __name__ == '__main__':
    unittest.main()
