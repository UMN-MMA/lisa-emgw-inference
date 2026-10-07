"""Posterior metrics. These do not establish MCMC convergence."""
import numpy as np


def intervals(samples, truth):
    x = np.asarray(samples, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if x.ndim != 2 or truth.shape != (x.shape[1],) or len(x) < 2:
        raise ValueError('Expected (draws, parameters) samples and matching truth')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(truth)):
        raise ValueError('Nonfinite samples must be investigated, not silently removed')
    lo, med, hi = np.quantile(x, [.05, .5, .95], axis=0)
    return {'lower_90': lo.tolist(), 'median': med.tolist(), 'upper_90': hi.tolist(),
            'width_90': (hi-lo).tolist(), 'displacement': (med-truth).tolist(),
            'contains_truth': ((lo <= truth) & (truth <= hi)).tolist()}


def wilson_interval(successes, total, z=1.959963984540054):
    if not isinstance(successes, (int, np.integer)) or not isinstance(total, (int, np.integer)):
        raise ValueError('Counts must be integers')
    if total <= 0 or not 0 <= successes <= total: raise ValueError('Invalid coverage counts')
    p = successes/total
    denominator = 1+z*z/total
    centre = (p+z*z/(2*total))/denominator
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return max(0., centre-half), min(1., centre+half)


def waveform_errors(gw_data, sampled_waveforms, truth_waveforms):
    """Input (draws,sources,channels,bins), returns component and total errors.

    Source association must already be resolved for component errors. Total error
    is invariant to source permutations. No posterior mean subtraction shortcut.
    """
    samples, truth = np.asarray(sampled_waveforms), np.asarray(truth_waveforms)
    if samples.ndim != 4 or samples.shape[1:] != truth.shape:
        raise ValueError('Waveform sample/truth shape mismatch')
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(truth)):
        raise ValueError('Waveforms must be finite')
    delta = samples-truth[None, ...]
    components = 4*gw_data.grid.df*np.sum(np.abs(delta)**2/gw_data.psd, axis=(-1,-2))
    total = 4*gw_data.grid.df*np.sum(np.abs(delta.sum(axis=1))**2/gw_data.psd, axis=(-1,-2))
    return components, total
