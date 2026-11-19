from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_autolab import read_json


class TestEurekaverseExample(unittest.TestCase):
    def test_generated_curriculum_beats_empty_levels(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)/'state'
            cmd = [sys.executable, str(ROOT/'rl_autolab.py'), 'run', '--config', str(ROOT/'configs/eurekaverse.json'),
                   '--state-dir', str(state)]
            first = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
            self.assertEqual(first.returncode, 3, first.stdout + first.stderr)
            self.assertIn('generate_level', (state/'proposal_request.md').read_text())
            (state/'proposal.json').write_text((ROOT/'examples/eurekaverse/proposal_curriculum.json').read_text())
            second = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            st = read_json(state/'state.json')
            base, cur = (row['metrics']['success_rate'] for row in st['history'])
            self.assertEqual(st['status'], 'TARGET_REACHED')
            self.assertGreater(cur, base + 0.1)


if __name__ == '__main__':
    unittest.main()
