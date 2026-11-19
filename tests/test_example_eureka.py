import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_autolab import read_json


class TestEurekaExample(unittest.TestCase):
    def test_sparse_to_shaped_reward(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)/'state'
            cmd = [sys.executable, str(ROOT/'rl_autolab.py'), 'run', '--config', str(ROOT/'configs/eureka.json'),
                   '--state-dir', str(state)]
            first = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
            self.assertEqual(first.returncode, 3, first.stdout + first.stderr)
            self.assertIn('r_success', (state/'proposal_request.md').read_text())  # reward reflection
            proposal = read_json(ROOT/'examples/eureka/proposal_shaped.json')
            (state/'proposal.json').write_text(json.dumps(proposal))
            second = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            st = read_json(state/'state.json')
            self.assertEqual(st['status'], 'TARGET_REACHED')
            self.assertLess(st['history'][0]['metrics']['success_rate'], 0.5)
            self.assertEqual(st['champion_code'], proposal['code'])


if __name__ == '__main__':
    unittest.main()
