"""CLI: validated analytic benchmark; explicit-input native experiment runner."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import os
import signal
import tempfile
import sys
import time
import numpy as np


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def versions():
    result = {'python':sys.version, 'source_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    for package in ('numpy','eryn','gbgpu','ellc'):
        try: result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: result[package] = None
    return result


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def doctor(native=False):
    report = {'versions':versions(), 'native_tests_requested':native}
    if native:
        scripts = {
            'ellc': "import ellc; print(ellc.lc(t_obs=[0.,.25],radius_1=.05,radius_2=.03,sbratio=.6,incl=88.,period=1.,shape_1='sphere',shape_2='sphere'))",
            'gbgpu': "from gbgpu.gbgpu import GBGPU; g=GBGPU(force_backend='cpu'); g.run_wave(1e-22,.0026,1e-17,0.,.1,1.4,.3,1.,.2,N=128,dt=15.,T=31558149.7635456); print(g.A.shape)"}
        for name, script in scripts.items():
            # Files avoid an unbounded communicate() when a native crash keeps pipes open.
            with tempfile.TemporaryFile(mode='w+') as stdout, tempfile.TemporaryFile(mode='w+') as stderr:
                child = subprocess.Popen([sys.executable,'-c',script],stdout=stdout,stderr=stderr,
                                         start_new_session=True)
                timed_out = False
                try:
                    child.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try: os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    try: child.wait(timeout=1)
                    except subprocess.TimeoutExpired: pass
                stdout.seek(0); stderr.seek(0)
                report[name] = {'returncode':child.returncode,
                                'ok':not timed_out and child.returncode==0,
                                'timed_out':timed_out,'pid':child.pid,
                                'stdout':stdout.read()[-1500:],'stderr':stderr.read()[-2000:]}
    return report


def native_run(config_path, arm, output):
    from .core import FrequencyGrid
    from .native import PhaseConvention, Binary, EMGeometry, GBGPUAdapter, EllcAdapter
    from .experiment import Coordinates, PhysicalLikelihood, observing_schedule, simulate, scale_to_snr
    from .core import GWData
    from .inference import sample_eryn
    from eryn.prior import ProbDistContainer, uniform_dist
    source = Path(config_path).resolve()
    config = json.loads(source.read_text())
    def resolve(value): return (source.parent/value).resolve()
    # Fail before native libraries load or any expensive calculation starts.
    if not config.get('psd_file'):
        raise ValueError('Supply psd_file with explicit frequencies and A/E one-sided PSD arrays')
    phase_spec = config.get('phase_convention')
    if not phase_spec or not phase_spec.get('validation_reference'):
        raise ValueError('Calibrate the GBGPU/ellc phase convention first; no guessed phase offset is supplied')
    initial_file = config.get('initial_files',{}).get(arm)
    if not initial_file: raise ValueError(f'Provide independently initialized ensemble for arm {arm}')
    if not config.get('prior_bounds'):
        raise ValueError('Freeze proper shared prior_bounds before inference')
    grid = FrequencyGrid(**config['grid'])
    with np.load(resolve(config['psd_file']),allow_pickle=False) as archive:
        psd, frequencies = archive['psd'],archive['frequencies']
    if frequencies.shape != (grid.count,) or not np.allclose(frequencies,grid.frequencies,rtol=0,atol=grid.df*1e-8):
        raise ValueError('PSD frequencies do not match absolute Fourier grid')
    phase = PhaseConvention(**phase_spec)
    gw_model = GBGPUAdapter(grid,phase,cadence_seconds=config.get('cadence_seconds',15.))
    em_model = EllcAdapter(config.get('exposure_order',12))
    binaries = [Binary(**x) for x in config['injection']['binaries']]
    if len(binaries)!=2: raise ValueError('This pilot runner requires target plus one neighbour')
    geometry = EMGeometry(**config['injection']['geometry'])
    empty = GWData(grid,np.zeros((2,grid.count),complex),psd)
    desired = config['injection'].get('individual_snrs')
    if desired is not None:
        if len(desired)!=2: raise ValueError('Two individual SNRs required')
        binaries = [scale_to_snr(b,s,gw_model,empty) for b,s in zip(binaries,desired)]
    times,exposures = observing_schedule(**config['observing_schedule'])
    gw,em,waves,flux = simulate(grid,psd,binaries,geometry,times,exposures,config['em_sigma'],
                              gw_model,em_model,config['gw_seed'],config['em_seed'])
    coordinates = Coordinates(config['frequency_reference'],grid.duration,
                               (binaries[0].longitude,binaries[0].sin_latitude),geometry)
    loglike = PhysicalLikelihood(gw,em,gw_model,em_model,coordinates,arm)
    # One common ordering: target6 + neighbour8 + EM7. All arms take the same prefixes.
    ndim = {'T':6,'G':14,'F':14,'J':21}[arm]
    bounds = np.asarray(config['prior_bounds'],dtype=float)
    if bounds.shape!=(21,2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:,1]<=bounds[:,0]):
        raise ValueError('prior_bounds must have shape (21,2) with finite increasing intervals')
    prior = ProbDistContainer({i:uniform_dist(*bounds[i]) for i in range(ndim)})
    initial = np.load(resolve(initial_file),allow_pickle=False)
    if initial.ndim != 3 or initial.shape[-1] != ndim:
        raise ValueError(f'Initial ensemble must have shape (temperatures, walkers, {ndim})')
    np.savez_compressed(output/'injection.npz',frequencies=grid.frequencies,psd=psd,
                        strain=gw.strain,times=times,exposures=exposures,
                        flux=em.flux,sigma=em.sigma,truth_waveforms=waves,truth_flux=flux)
    dump(output/'input_config.json',config)
    chain, diagnostics = sample_eryn(loglike,prior,initial,
                                    config['sampling']['steps'],config['sampling']['burn'],
                                    config['initialization_seeds'][arm],coordinates.periodic(arm))
    np.savez_compressed(output/'posterior.npz',chain=chain)
    np.savez_compressed(output/'injection.npz',frequencies=grid.frequencies,psd=psd,
                        strain=gw.strain,times=times,exposures=exposures,
                        flux=em.flux,sigma=em.sigma,truth_waveforms=waves,truth_flux=flux)
    # Summarize only non-circular, identifiable coordinates; preserve the full chain.
    from .metrics import intervals, waveform_errors
    flat=chain.reshape(-1,ndim)
    indices=[0,1,2,4] if arm=='T' else [0,1,2,4,6,7,8,10]
    names=['lnA_target','f_target_hz','fdot_target','abs_cosi_target']
    if arm!='T': names += ['lnA_neighbour','f_neighbour_hz','fdot_neighbour','abs_cosi_neighbour']
    def scalars(x):
        z=x[:,indices].copy()
        for start in range(0,len(indices),4):
            z[:,start+1]=coordinates.fref+z[:,start+1]/grid.duration
            z[:,start+2]/=grid.duration**2
            z[:,start+3]=np.abs(z[:,start+3])
        return z
    truth=[]
    for binary in binaries[:1 if arm=='T' else 2]:
        truth.extend([np.log(binary.amplitude),binary.frequency,binary.fdot,abs(binary.cosi)])
    summary=intervals(scalars(flat),np.array(truth))
    summary['names']=names
    summary['source_association']='conditional_on_parameter_labels; inspect ambiguous cases before population summaries'
    count=config.get('diagnostic_waveform_draws',0)
    if not isinstance(count,int) or count<0: raise ValueError('diagnostic_waveform_draws must be nonnegative integer')
    if count:
        selection=np.random.default_rng(9001).choice(len(flat),size=min(count,len(flat)),replace=False)
        predictions=[]
        for index in selection:
            sources,_=coordinates.decode(flat[index],arm)
            predicted=gw_model(sources)
            if arm=='T': predicted=np.concatenate([predicted,np.zeros_like(predicted)],axis=0)
            predictions.append(predicted)
        component_errors,total_errors=waveform_errors(gw,np.array(predictions),waves)
        np.savez_compressed(output/'waveform_errors.npz',posterior_indices=selection,
                            components=component_errors,total=total_errors)
    dump(output/'summary.json',summary)
    dump(output/'diagnostics.json',diagnostics)
    dump(output/'scaled_injection.json',{'binaries':[asdict(b) for b in binaries],'geometry':asdict(geometry)})
    return {'kind':'physical_native_pilot','arm':arm,'convergence':'requires_independent_assessment',
            'input_config_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'data_sha256':hashlib.sha256(gw.strain.tobytes()+em.flux.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    check = commands.add_parser('doctor'); check.add_argument('--native',action='store_true')
    bench = commands.add_parser('benchmark')
    bench.add_argument('--overlap',type=float,default=.8)
    bench.add_argument('--seed',type=int,default=42)
    bench.add_argument('--draws',type=int,default=20000)
    bench.add_argument('--sampler',choices=['analytic','eryn'],default='analytic')
    bench.add_argument('--steps',type=int,default=1000); bench.add_argument('--burn',type=int,default=500)
    bench.add_argument('--out',required=True,type=Path)
    native = commands.add_parser('run-native')
    native.add_argument('--config',required=True,type=Path)
    native.add_argument('--arm',choices=['T','G','F','J'],required=True)
    native.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    if args.command=='doctor':
        print(json.dumps(doctor(args.native),indent=2)); return
    output = new_output(args.out)
    started = time.monotonic()
    metadata = {'status':'running','versions':versions(),'command':vars(args).copy()}
    metadata['command'] = {k:str(v) if isinstance(v,Path) else v for k,v in metadata['command'].items()}
    dump(output/'run.json',metadata)
    try:
        if args.command=='benchmark':
            from .benchmark import run_benchmark
            report,arrays = run_benchmark(args.overlap,args.seed,args.draws,args.sampler,args.steps,args.burn)
            np.savez_compressed(output/'samples.npz',**arrays)
            dump(output/'summary.json',report)
            metadata['kind'] = report['kind']
        else:
            metadata.update(native_run(args.config,args.arm,output))
        metadata['status'] = 'completed_execution_not_convergence_certification'
    except Exception as exc:
        metadata.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata['elapsed_seconds'] = time.monotonic()-started
        dump(output/'run.json',metadata)
    print(json.dumps(metadata,indent=2))


if __name__=='__main__': main()
