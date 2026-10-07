"""Physical fixed-count experiment definitions and reproducible data generation."""
from dataclasses import replace
import numpy as np
from .core import GWData, Photometry, complex_noise
from .native import Binary, EMGeometry


class Coordinates:
    """Dimensionless sampling basis.

    Target: lnA, (f-fref)T, fdot*T², orbital phase cycles, cosi, psi.
    Neighbour: same six plus longitude and sin(latitude).
    J adds r1,r2,lnJ,u1,u2,ln(flux_scale),ln(jitter).
    Target sky is fixed identically in every arm. Source-count is known.
    """
    def __init__(self, frequency_reference, duration, target_sky, fixed_geometry):
        self.fref, self.duration = frequency_reference, duration
        self.target_sky, self.fixed_geometry = target_sky, fixed_geometry

    def decode_binary(self, x, sky):
        t = self.duration
        return Binary(float(np.exp(x[0])), self.fref+x[1]/t, x[2]/t**2,
                      x[3], x[4], x[5], *sky)

    def decode(self, vector, arm):
        if arm not in ('T', 'G', 'F', 'J'): raise ValueError('Unknown comparison arm')
        x = np.asarray(vector, dtype=float)
        size = {'T': 6, 'G': 14, 'F': 14, 'J': 21}[arm]
        if x.shape != (size,): raise ValueError(f'{arm} requires {size} coordinates')
        if not np.all(np.isfinite(x)): raise ValueError('Nonfinite coordinates')
        with np.errstate(over='ignore'):
            sources = [self.decode_binary(x[:6], self.target_sky)]
            if arm != 'T': sources.append(self.decode_binary(x[6:12], x[12:14]))
            em = self.fixed_geometry
            if arm == 'J':
                z = x[14:]
                em = EMGeometry(z[0], z[1], np.exp(z[2]), z[3], z[4], np.exp(z[5]), np.exp(z[6]))
        return sources, em

    def periodic(self, arm):
        periodic = {3: 1.0, 5: np.pi}
        if arm != 'T': periodic.update({9: 1.0, 11: np.pi, 12: 2*np.pi})
        return periodic


class PhysicalLikelihood:
    def __init__(self, gw_data, photometry, gw_model, em_model, coordinates, arm):
        self.gw, self.em = gw_data, photometry
        self.gw_model, self.em_model = gw_model, em_model
        self.coordinates, self.arm = coordinates, arm
        if arm not in ('T', 'G', 'F', 'J'): raise ValueError('Invalid arm')

    def __call__(self, vectors):
        vectors = np.atleast_2d(vectors)
        likelihood = np.empty(len(vectors))
        for i, vector in enumerate(vectors):
            try:
                sources, geometry = self.coordinates.decode(vector, self.arm)
            except ValueError:
                likelihood[i] = -np.inf
                continue
            # Library/runtime failures intentionally propagate; they are not prior rejections.
            waves = self.gw_model(sources)
            ll = self.gw.loglike(waves.sum(axis=0))
            if self.arm in ('F', 'J'):
                prediction = self.em_model(sources[0], geometry, self.em.times, self.em.exposure)
                ll += self.em.loglike(prediction, geometry.jitter)
            likelihood[i] = ll
        return likelihood


def observing_schedule(visit_days, hours=2., cadence=15., exposure=10.):
    if not 0 < exposure <= cadence or hours <= 0: raise ValueError('Invalid observing schedule')
    if not visit_days or len(set(visit_days)) != len(visit_days): raise ValueError('Unique visits required')
    centres = np.arange(0., hours*3600-exposure+1e-10, cadence)+exposure/2
    times = np.sort(np.concatenate([day*86400+centres for day in visit_days]))
    if np.any(np.diff(times) < exposure): raise ValueError('Visits or exposures overlap')
    return times, np.full(times.size, exposure)


def simulate(grid, psd, binaries, geometry, times, exposures, sigma,
             gw_model, em_model, gw_seed, em_seed):
    """Generate one realization to be REUSED across T/G/F/J."""
    truth_waveforms = gw_model(binaries)
    empty = GWData(grid, np.zeros((2, grid.count), complex), psd)
    if truth_waveforms.shape != (len(binaries), 2, grid.count):
        raise ValueError('Native waveform model returned unexpected shape')
    noise = complex_noise(empty.psd, grid.df, np.random.default_rng(gw_seed))
    gw = GWData(grid, truth_waveforms.sum(axis=0)+noise, psd)
    true_flux = em_model(binaries[0], geometry, times, exposures)
    sigma = np.broadcast_to(sigma, times.shape).copy()
    clean = Photometry(times, exposures, true_flux, sigma)
    variance = clean.sigma**2+geometry.jitter**2
    flux = true_flux+np.random.default_rng(em_seed).normal(size=times.shape)*np.sqrt(variance)
    return gw, Photometry(times, exposures, flux, sigma), truth_waveforms, true_flux


def scale_to_snr(binary, desired_snr, gw_model, gw_data):
    if not np.isfinite(desired_snr) or desired_snr <= 0: raise ValueError('Positive SNR required')
    waveform = gw_model([binary])[0]
    norm = np.sqrt(gw_data.inner(waveform, waveform))
    if norm <= 0: raise ValueError('Signal has no support in this data window')
    return replace(binary, amplitude=binary.amplitude*desired_snr/norm)
