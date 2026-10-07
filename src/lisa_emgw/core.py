"""Explicit one-sided-PSD convention for positive-frequency A/E data."""
from dataclasses import dataclass
import numpy as np


def finite_array(value, name, dtype=float):
    value = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be finite')
    return value


@dataclass(frozen=True)
class FrequencyGrid:
    first_bin: int
    count: int
    duration: float

    def __post_init__(self):
        if not isinstance(self.first_bin, (int, np.integer)) or self.first_bin <= 0:
            raise ValueError('first_bin must be a positive integer; DC is excluded')
        if not isinstance(self.count, (int, np.integer)) or self.count < 2:
            raise ValueError('count must be an integer >= 2')
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError('duration must be finite and positive')

    @property
    def df(self): return 1.0/self.duration

    @property
    def frequencies(self): return np.arange(self.first_bin, self.first_bin+self.count)*self.df


def align_waveforms(channels, starts, grid):
    """(sources, channels, local bins) -> (sources, channels, absolute data bins)."""
    channels = finite_array(channels, 'waveforms', complex)
    starts = finite_array(starts, 'start indices')
    if channels.ndim != 3 or starts.shape != (channels.shape[0],):
        raise ValueError('Expected channels (sources, channels, bins) and starts (sources,)')
    if np.any(starts != np.rint(starts)):
        raise ValueError('Waveform start indices must be integers')
    output = np.zeros((*channels.shape[:2], grid.count), complex)
    for i, start in enumerate(starts.astype(int)):
        lo = max(grid.first_bin, start)
        hi = min(grid.first_bin+grid.count, start+channels.shape[-1])
        if hi > lo:
            output[i, :, lo-grid.first_bin:hi-grid.first_bin] = channels[i, :, lo-start:hi-start]
    return output


@dataclass(frozen=True)
class GWData:
    grid: FrequencyGrid
    strain: np.ndarray
    psd: np.ndarray

    def __post_init__(self):
        strain = finite_array(self.strain, 'strain', complex).copy()
        psd = finite_array(self.psd, 'PSD').copy()
        if strain.shape != (2, self.grid.count) or psd.shape != strain.shape:
            raise ValueError('A/E strain and one-sided PSD must both have shape (2, bins)')
        if np.any(psd <= 0): raise ValueError('PSD must be positive')
        strain.flags.writeable = psd.flags.writeable = False
        object.__setattr__(self, 'strain', strain)
        object.__setattr__(self, 'psd', psd)

    def inner(self, x, y, noise_scales=None):
        x, y = finite_array(x, 'x', complex), finite_array(y, 'y', complex)
        if x.shape != self.strain.shape or y.shape != x.shape:
            raise ValueError('Inner product arrays must match the data')
        return float(4*self.grid.df*np.real(np.sum(x.conj()*y/self.scaled_psd(noise_scales))))

    def scaled_psd(self, scales=None):
        if scales is None: return self.psd
        scales = finite_array(scales, 'channel PSD scales')
        if scales.shape != (2,) or np.any(scales <= 0):
            raise ValueError('Expected two positive PSD multipliers')
        return self.psd*scales[:, None]

    def loglike(self, model, noise_scales=None):
        model = np.asarray(model, dtype=complex)
        if model.shape != self.strain.shape: raise ValueError('Model shape differs from data')
        if not np.all(np.isfinite(model)): return -np.inf
        psd = self.scaled_psd(noise_scales)
        residual = self.strain-model
        with np.errstate(over='ignore', invalid='ignore'):
            value = -2*self.grid.df*np.sum(np.abs(residual)**2/psd)
            value -= np.sum(np.log(np.pi*psd/(2*self.grid.df)))
        return float(value) if np.isfinite(value) else -np.inf


def complex_noise(psd, df, rng):
    """Each quadrature has variance S/(4 df), hence E|n|²=S/(2 df)."""
    psd = finite_array(psd, 'PSD')
    if np.any(psd <= 0) or not np.isfinite(df) or df <= 0:
        raise ValueError('Positive PSD and df required')
    return np.sqrt(psd/(4*df))*(rng.normal(size=psd.shape)+1j*rng.normal(size=psd.shape))


@dataclass(frozen=True)
class Photometry:
    times: np.ndarray
    exposure: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray

    def __post_init__(self):
        shape = np.asarray(self.times).shape
        if len(shape) != 1 or not shape[0]: raise ValueError('Expected nonempty time vector')
        for name in ('times', 'exposure', 'flux', 'sigma'):
            value = finite_array(getattr(self, name), name).copy()
            if value.shape != shape: raise ValueError('Photometry vectors must have identical shapes')
            if name in ('exposure', 'sigma') and np.any(value <= 0):
                raise ValueError(f'{name} must be positive')
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def loglike(self, prediction, jitter=0.0):
        prediction = np.asarray(prediction)
        if prediction.shape != self.flux.shape: raise ValueError('Flux prediction shape mismatch')
        if not np.isfinite(jitter) or jitter < 0: return -np.inf
        if not np.all(np.isfinite(prediction)): return -np.inf
        with np.errstate(over='ignore', invalid='ignore'):
            variance = self.sigma**2+jitter**2
            ll = -.5*np.sum((self.flux-prediction)**2/variance+np.log(2*np.pi*variance))
        return float(ll) if np.isfinite(ll) else -np.inf


class ModelCache:
    """Explicit transactions: proposing must never mutate the accepted state."""
    def __init__(self, waveforms):
        self._waves = finite_array(waveforms, 'waveforms', complex).copy()
        if self._waves.ndim != 3: raise ValueError('Expected (sources, channels, bins)')
        self._total = self._waves.sum(axis=0)

    @property
    def total(self): return self._total.copy()

    def proposed_total(self, index, replacement):
        replacement = finite_array(replacement, 'replacement', complex)
        if replacement.shape != self._total.shape: raise ValueError('Replacement shape mismatch')
        return self._total-self._waves[index]+replacement

    def accept(self, index, replacement):
        total = self.proposed_total(index, replacement)
        self._waves[index] = replacement
        self._total = total

    def full_total(self): return self._waves.sum(axis=0)


def shared_orbital_phase(times, f_gw, fdot_gw, phase_cycles, reference_time=0.):
    times = finite_array(times, 'times')
    if not np.all(np.isfinite([f_gw, fdot_gw, phase_cycles, reference_time])) or f_gw <= 0:
        raise ValueError('Invalid orbital parameters')
    dt = times-reference_time
    if np.any(f_gw+fdot_gw*dt <= 0): raise ValueError('Orbital frequency becomes nonpositive')
    return phase_cycles + .5*f_gw*dt + .25*fdot_gw*dt**2


def exposure_average(times, exposures, flux_at_time, order=12):
    times, exposures = finite_array(times, 'times'), finite_array(exposures, 'exposures')
    if times.ndim != 1 or times.shape != exposures.shape or np.any(exposures <= 0):
        raise ValueError('Time and positive-exposure vectors must match')
    if not isinstance(order, int) or order < 2: raise ValueError('Quadrature order must be >=2')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    sample_times = times[:, None]+.5*exposures[:, None]*nodes
    values = np.asarray(flux_at_time(sample_times.reshape(-1))).reshape(sample_times.shape)
    return .5*np.sum(values*weights, axis=1)
