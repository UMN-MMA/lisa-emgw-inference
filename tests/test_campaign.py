import json
import unittest
from collections import defaultdict
from pathlib import Path
from scripts.prepare_campaign import make_rows


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / 'experiments' / 'campaign.json').read_text())

    def test_counts(self):
        for stage, count in [('pilot', 144), ('main', 7680)]:
            self.assertEqual(len(list(make_rows(self.config, stage))), count)

    def test_comparison_arms_share_data_not_initialization(self):
        groups = defaultdict(list)
        for row in make_rows(self.config, 'pilot'):
            groups[row['configuration_id'], row['realization']].append(row)
        for rows in groups.values():
            for key in ['gw_noise_seed', 'em_noise_seed', 'geometry_seed']:
                self.assertEqual(len({r[key] for r in rows}), 1)
            self.assertEqual(len({r['initialization_seed'] for r in rows}), 4)

    def test_fixed_truth_independent_noise_and_reproducibility(self):
        rows = list(make_rows(self.config, 'pilot'))
        self.assertEqual(rows, list(make_rows(self.config, 'pilot')))
        groups = defaultdict(list)
        for row in rows:
            if row['arm'] == 'G': groups[row['configuration_id']].append(row)
        for group in groups.values():
            self.assertEqual(len({r['geometry_seed'] for r in group}), 1)
            self.assertEqual(len({r['gw_noise_seed'] for r in group}), 3)
            self.assertEqual(len({r['em_noise_seed'] for r in group}), 3)


if __name__ == '__main__':
    unittest.main()
