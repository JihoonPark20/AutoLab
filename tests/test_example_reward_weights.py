import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_autolab import read_json


class TestRewardWeightsExample(unittest.TestCase):
    def test_two_step_weight_tuning(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)/'state'
            cmd = [sys.executable, str(ROOT/'rl_autolab.py'), 'run', '--config', str(ROOT/'configs/reward_weights.json'),
                   '--state-dir', str(state)]
            for proposal in (None, 'proposal_1_progress.json', 'proposal_2_step.json'):
                if proposal:
                    (state/'proposal.json').write_text((ROOT/'examples/reward_weights'/proposal).read_text())
                got = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
            self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
            st = read_json(state/'state.json')
            scores = [row['metrics']['success_rate'] for row in st['history']]
            self.assertEqual(st['status'], 'TARGET_REACHED')
            self.assertTrue(scores[0] < 0.5 < scores[1] < scores[2], scores)
            self.assertEqual(st['champion_params'], {'w_goal': 1.0, 'w_progress': 0.05, 'w_step': -0.05})


if __name__ == '__main__':
    unittest.main()
