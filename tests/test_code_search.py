import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_autolab import check_code, generate_next_params, init_state, read_json
from tests.fixtures import toy_config

SPEC = {'function': 'compute_reward', 'allowed_imports': ['math'], 'max_bytes': 400}
GOOD = 'import math\ndef compute_reward(obs):\n    return math.tanh(1.0), {}\n'

TRAIN = '''import argparse, importlib.util, json, pathlib
p = argparse.ArgumentParser(); p.add_argument('--run-dir'); p.add_argument('--code-file'); a = p.parse_args()
spec = importlib.util.spec_from_file_location('candidate', a.code_file)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
value, parts = mod.compute_reward({})
run = pathlib.Path(a.run_dir)
(run / 'train_metrics.jsonl').write_text(json.dumps({f'r_{k}': v for k, v in parts.items()}) + '\\n')
(run / 'ckpt.json').write_text(json.dumps({'value': value}))
(run / 'checkpoint_path.txt').write_text(str((run / 'ckpt.json').resolve()))
'''
EVAL = '''import argparse, json, pathlib
p = argparse.ArgumentParser(); p.add_argument('--run-dir'); p.add_argument('--checkpoint'); p.add_argument('--seeds'); a = p.parse_args()
value = json.loads(pathlib.Path(a.checkpoint).read_text())['value']
seeds = [int(s) for s in a.seeds.split(',')]
pathlib.Path(a.run_dir, 'eval.json').write_text(json.dumps({'evaluation_seeds': seeds, 'metrics': {'score': value}}))
'''


class TestCodeSearch(unittest.TestCase):
    def test_static_check(self):
        check_code(GOOD, SPEC)
        bad = {
            'import os\ndef compute_reward(obs): return 0, {}': 'import',
            'from . import x\ndef compute_reward(obs): return 0, {}': 'import',
            'def compute_reward(obs): return open("/etc/passwd"), {}': 'blocked',
            'def compute_reward(obs): return ().__class__, {}': 'dunder',
            'def compute_reward(obs): return __builtins__, {}': 'blocked',
            'def other(obs): return 0, {}': 'top-level',
            'def compute_reward(obs) return': 'syntax',
            'x = 1\n' * 100 + 'def compute_reward(obs): return 0, {}': 'max_bytes',
        }
        for code, why in bad.items():
            with self.subTest(why=why), self.assertRaisesRegex(ValueError, why):
                check_code(code, SPEC)

    def test_code_needs_code_section(self):
        cfg = toy_config()
        cfg['backend'] = 'manual'
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td/'proposal.json').write_text(json.dumps({'code': GOOD}))
            st = init_state(cfg)
            self.assertFalse(generate_next_params(st, cfg, td, td/'state.json'))
            self.assertEqual(st['status'], 'NEEDS_REVIEW')
            self.assertIn('no code section', st['pause_reason'])

    def test_code_candidate_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proj = td/'project'; proj.mkdir()
            (proj/'train.py').write_text(TRAIN)
            (proj/'evaluate.py').write_text(EVAL)
            (proj/'env.py').write_text('# environment source shown to the planner\n')
            (td/'initial.py').write_text('def compute_reward(obs):\n    return 0.1, {"bonus": 0.1}\n')
            before = {x.name: x.read_bytes() for x in proj.iterdir()}
            cfg = {'name': 'code_fixture', 'backend': 'manual', 'max_cycles': 4, 'poll_seconds': 0.05,
                   'parameters': {'lr': {'value': 0.5, 'min': 0.1, 'max': 1.0, 'max_delta': 0.1}},
                   'code': {'initial': 'initial.py', 'role': 'training reward', 'context_files': ['env.py']},
                   'train': {'command': ['{python}', 'train.py', '--run-dir', '{run_dir}', '--code-file', '{code_file}'],
                             'timeout_seconds': 30},
                   'eval': {'command': ['{python}', 'evaluate.py', '--run-dir', '{run_dir}', '--checkpoint', '{checkpoint}',
                                        '--seeds', '{fixed_seeds}'], 'timeout_seconds': 30, 'fixed_seeds': [1, 2]},
                   'decision': {'primary_metric': 'score', 'maximize': True, 'target_score': 0.9, 'min_improvement': 0.01},
                   'limits': {}, 'project': {'root': 'project'}}
            (td/'cfg.json').write_text(json.dumps(cfg))
            state = td/'state'
            cmd = [sys.executable, str(ROOT/'rl_autolab.py'), 'run', '--config', str(td/'cfg.json'), '--state-dir', str(state)]
            first = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
            self.assertEqual(first.returncode, 3, first.stdout + first.stderr)
            prompt = (state/'proposal_request.md').read_text()
            for needle in ('CURRENT_BEST_CODE', 'training reward', 'r_bonus', 'environment source shown'):
                self.assertIn(needle, prompt)
            new = 'def compute_reward(obs):\n    return 0.95, {"bonus": 0.95}\n'
            (state/'proposal.json').write_text(json.dumps({'hypothesis': 'h', 'code': new, 'expected_effect': 'e'}))
            second = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            st = read_json(state/'state.json')
            self.assertEqual(st['status'], 'TARGET_REACHED')
            self.assertEqual(st['champion_code'], new)
            self.assertEqual((state/'runs/cycle_001/candidate.py').read_text(), new)
            self.assertEqual({x.name: x.read_bytes() for x in proj.iterdir()}, before)


if __name__ == '__main__':
    unittest.main()
