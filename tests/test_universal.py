import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_autolab import read_json, project_root, project_python, run_manager, validate_config


class TestUniversal(unittest.TestCase):
    def test_project_is_resolved_outside_manager(self):
        cfg_file = ROOT/'configs/external_project_demo.json'
        cfg = read_json(cfg_file)
        self.assertEqual(project_root(cfg, cfg_file), ROOT/'examples/toy_project')
        self.assertEqual(project_python(cfg, ROOT/'examples/toy_project'), sys.executable)
        validate_config(cfg)

    def test_external_project_remains_unmodified(self):
        cfg_file = ROOT/'configs/external_project_demo.json'
        root = ROOT/'examples/toy_project'
        before = {x.name: x.read_bytes() for x in root.iterdir() if x.is_file()}
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)/'state'
            completed = subprocess.run([sys.executable, str(ROOT/'rl_autolab.py'), 'run',
                                        '--config', str(cfg_file), '--state-dir', str(state), '--cycles', '2'],
                                       capture_output=True, text=True, timeout=35)
            self.assertEqual(completed.returncode, 0, completed.stdout+'\n'+completed.stderr)
            self.assertEqual(read_json(state/'state.json')['next_cycle'], 2)
            self.assertEqual({x.name: x.read_bytes() for x in root.iterdir() if x.is_file()}, before)
            self.assertTrue((state/'runs/cycle_000/seed_40/eval.json').is_file())
            self.assertTrue((state/'runs/cycle_001/seed_41/eval.json').is_file())

    def test_enforces_external_output(self):
        cfg_file = ROOT/'configs/external_project_demo.json'
        with self.assertRaisesRegex(ValueError, 'OUTSIDE'):
            run_manager(cfg_file, ROOT/'examples/toy_project/runs/test', 1, .2)


if __name__ == '__main__':
    unittest.main()
