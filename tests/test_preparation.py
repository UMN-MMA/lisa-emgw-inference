import json
from pathlib import Path
import numpy as np
from scripts.make_native_config import make_config


def test_native_configuration_needs_no_generated_manifest():
    config = make_config('pilot-000', 0)
    example = json.loads((Path(__file__).resolve().parents[1]/'configs/pilot-000.example.json').read_text())
    assert config == example
    assert config['psd_file'] is None
    assert config['phase_convention']['validation_reference'] == ''
    assert np.asarray(config['prior_bounds']).shape == (21, 2)
