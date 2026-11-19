from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_autolab import read_json


class TestDrEurekaExample(unittest.TestCase):
    def test_domain_randomization_closes_sim_to_real_gap(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)/'state'
            cmd = [sys.executable, str(ROOT/'rl_autolab.py'), 'run', '--config', str(ROOT/'configs/dreureka.json'),
                   '--state-dir', str(state)]
            first = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
            self.assertEqual(first.returncode, 3, first.stdout + first.stderr)
            (state/'proposal.json').write_text((ROOT/'examples/dreureka/proposal_dr.json').read_text())
            second = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            st = read_json(state/'state.json')
            base, dr = (row['metrics'] for row in st['history'])
            self.assertEqual(st['status'], 'TARGET_REACHED')
            self.assertGreaterEqual(base['sim_success_rate'], 0.9)  # the gap is sim-to-real, not a broken policy
            self.assertLess(base['success_rate'], 0.7)
            self.assertGreater(dr['success_rate'], 0.85)


if __name__ == '__main__':
    unittest.main()
