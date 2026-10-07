"""Generate a deterministic experiment manifest. Does not run scientific inference."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments"
AXES = ('separation_bins', 'neighbour_snr_ratio', 'eclipse_impact', 'target_snr', 'em_sigma')


def seed_for(base, *parts):
    # Stable across Python processes; do not use the randomized built-in hash().
    payload = json.dumps([base, *parts], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')


def make_rows(config, stage):
    design = config['stages'][stage]
    base = config['base_seed']
    for index, values in enumerate(itertools.product(*(design[a] for a in AXES))):
        cid = f'{stage}-{index:03d}'
        settings = dict(zip(AXES, values))
        # One physical truth per configuration; noise changes independently.
        sign = 1  # mirrored offsets are a separate robustness experiment
        for realization in range(design['noise_realizations']):
            common = dict(configuration_id=cid, realization=realization, **settings,
                          separation_sign=sign,
                          delta_f_hz=sign*settings['separation_bins']/config['shared']['gw_duration_seconds'],
                          geometry_seed=seed_for(base, cid, 'geometry'),
                          gw_noise_seed=seed_for(base, cid, realization, 'gw'),
                          em_noise_seed=seed_for(base, cid, realization, 'em'))
            for arm in config['arms']:
                yield dict(common, arm=arm,
                           initialization_seed=seed_for(base, cid, realization, arm, 'init'),
                           status='planned_not_run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'main'), default='pilot')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raw = (ROOT/'campaign.json').read_bytes()
    config = json.loads(raw)
    rows = list(make_rows(config, args.stage))
    output = args.output or ROOT/f'{args.stage}_manifest.csv'
    if output.exists():
        raise SystemExit(f'Refusing to overwrite existing manifest: {output}')
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(stage=args.stage, configurations=len({r['configuration_id'] for r in rows}),
                    posterior_fits_before_repeats=len(rows),
                    campaign_sha256=hashlib.sha256(raw).hexdigest(),
                    scientific_runs_executed=0,
                    readiness=config['status'])
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
