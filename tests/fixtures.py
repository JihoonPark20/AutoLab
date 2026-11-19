from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_autolab import read_json


def toy_config() -> dict:
    """Toy-project config with an absolute root, so tests can copy it anywhere."""
    cfg = read_json(ROOT/'configs/external_project_demo.json')
    cfg['project']['root'] = str(ROOT/'examples/toy_project')
    return cfg
