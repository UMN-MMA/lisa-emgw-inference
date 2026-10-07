"""Translate ONE pilot configuration into an editable native-run specification.

Creates no signals and runs no MCMC. PSD, phase calibration and initialization
files must be supplied before run-native can execute. Priors are draft bounds.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1] / "experiments"

if __package__:
    from .prepare_campaign import make_rows
else:
    from prepare_campaign import make_rows


def make_config(configuration_id,realization):
    cfg=json.loads((ROOT/'campaign.json').read_text())
    selected=[r for r in make_rows(cfg,'pilot')
              if r['configuration_id']==configuration_id and r['realization']==realization]
    if len(selected)!=4: raise ValueError('Expected one manifest row for each of T/G/F/J')
    row=next(r for r in selected if r['arm']=='G')
    shared=cfg['shared'];t=shared['gw_duration_seconds'];f=shared['frequency_hz']
    G,c,MSUN,RSUN=6.67430e-11,299792458.,1.988409870698051e30,6.957e8
    m1,m2=[m*MSUN for m in shared['illustrative_injection_masses_msun']]
    a=(G*(m1+m2)*(1/(math.pi*f))**2)**(1/3)
    radii=[r*RSUN/a for r in shared['illustrative_injection_radii_rsun']]
    chirp=(m1*m2)**(.6)/(m1+m2)**(.2)
    fdot=96/5*math.pi**(8/3)*(G*chirp/c**3)**(5/3)*f**(11/3)
    rng=np.random.default_rng(int(row['geometry_seed']))
    target=dict(amplitude=1e-22,frequency=f,fdot=fdot,orbital_phase=.17,
                cosi=float(row['eclipse_impact'])*sum(radii),polarization=.8,
                longitude=shared['target_sky_longitude_rad'],
                sin_latitude=math.sin(shared['target_sky_latitude_rad']))
    neighbour=dict(amplitude=1e-22,frequency=f+float(row['delta_f_hz']),fdot=fdot,
                   orbital_phase=float(rng.uniform(0,1)),cosi=float(rng.uniform(-1,1)),
                   polarization=float(rng.uniform(0,math.pi)),longitude=float(rng.uniform(0,2*math.pi)),
                   sin_latitude=float(rng.uniform(-1,1)))
    six=[[-60,-40],[-20,20],[-100,100],[0,1],[-1,1],[0,math.pi]]
    bounds=six+six+[[0,2*math.pi],[-1,1]]+[
        [.005,.25],[.005,.25],[math.log(.05),math.log(5)],[0,1],[0,1],
        [math.log(.95),math.log(1.05)],[-16,-4]]
    snr=float(row['target_snr'])
    return dict(status='draft_config_not_native_validated',configuration_id=configuration_id,
                realization=realization,
                grid=dict(first_bin=int(f*t)-512,count=1024,duration=t),
                frequency_reference=f,psd_file=None,
                phase_convention=dict(offset_radians=None,sign=None,validation_reference=''),
                cadence_seconds=15.,exposure_order=12,
                injection=dict(binaries=[target,neighbour],individual_snrs=[snr,snr*float(row['neighbour_snr_ratio'])],
                               geometry=dict(radius_1=radii[0],radius_2=radii[1],surface_brightness_ratio=.6,
                                             limb_1=.3,limb_2=.4,flux_scale=1.,jitter=0.)),
                observing_schedule=dict(visit_days=shared['em_visit_start_days'],hours=2.,cadence=15.,exposure=10.),
                em_sigma=float(row['em_sigma']),gw_seed=int(row['gw_noise_seed']),em_seed=int(row['em_noise_seed']),
                initialization_seeds={r['arm']:int(r['initialization_seed']) for r in selected},
                initial_files={arm:None for arm in ['T','G','F','J']},
                prior_bounds=bounds,prior_status='draft_bounds_require_sensitivity_review',
                sampling=dict(steps=1000,burn=500),
                diagnostic_waveform_draws=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',default='pilot-000')
    parser.add_argument('--realization',type=int,default=0)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    spec=make_config(args.configuration,args.realization)
    with args.out.open('x') as stream: json.dump(spec,stream,indent=2);stream.write('\n')
    print(f'Created editable configuration: {args.out}. No inference performed.')
