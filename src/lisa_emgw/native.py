"""Optional native adapters. Importing this module does not load native libraries.

PhaseConvention must be independently calibrated to GBGPU/ellc conventions.
The code implements the mapping, but does not establish its physical correctness.
"""
from dataclasses import dataclass
import numpy as np
from .core import align_waveforms, shared_orbital_phase, exposure_average


@dataclass(frozen=True)
class PhaseConvention:
    offset_radians: float
    sign: int
    validation_reference: str

    def __post_init__(self):
        if self.sign not in (-1, 1) or not np.isfinite(self.offset_radians):
            raise ValueError('Phase convention requires finite offset and sign +/-1')
        if not self.validation_reference.strip():
            raise ValueError('A phase-calibration test/report reference is required; do not guess the offset')

    def gw_phase(self, orbital_phase_cycles):
        return (self.sign*4*np.pi*orbital_phase_cycles+self.offset_radians) % (2*np.pi)


@dataclass(frozen=True)
class Binary:
    amplitude: float
    frequency: float
    fdot: float
    orbital_phase: float
    cosi: float
    polarization: float
    longitude: float
    sin_latitude: float

    def __post_init__(self):
        if not np.all(np.isfinite(list(self.__dict__.values()))): raise ValueError('Binary parameters must be finite')
        if self.amplitude <= 0 or self.frequency <= 0: raise ValueError('Positive amplitude and frequency required')
        if abs(self.cosi) > 1 or abs(self.sin_latitude) > 1: raise ValueError('Invalid orientation')


@dataclass(frozen=True)
class EMGeometry:
    radius_1: float
    radius_2: float
    surface_brightness_ratio: float
    limb_1: float
    limb_2: float
    flux_scale: float = 1.0
    jitter: float = 0.0

    def __post_init__(self):
        if not np.all(np.isfinite(list(self.__dict__.values()))): raise ValueError('Geometry must be finite')
        if min(self.radius_1, self.radius_2, self.surface_brightness_ratio, self.flux_scale) <= 0:
            raise ValueError('Radii, brightness ratio and flux scale must be positive')
        if self.radius_1+self.radius_2 >= 1: raise ValueError('Spherical stars overlap')
        if not 0 <= self.limb_1 <= 1 or not 0 <= self.limb_2 <= 1 or self.jitter < 0:
            raise ValueError('Invalid linear limb darkening or jitter')


class GBGPUAdapter:
    def __init__(self, grid, phase_convention, cadence_seconds=15., waveform_bins=None, backend=None):
        self.grid, self.phase_convention = grid, phase_convention
        if cadence_seconds <= 0 or grid.frequencies[-1] >= .5/cadence_seconds:
            raise ValueError('Data window must be strictly below Nyquist')
        self.cadence = cadence_seconds
        self.waveform_bins = waveform_bins or 2**int(np.ceil(np.log2(2*grid.count)))
        if backend is None:
            from gbgpu.gbgpu import GBGPU
            backend = GBGPU(force_backend='cpu')
        self.backend = backend

    def __call__(self, binaries):
        if not binaries: raise ValueError('At least one binary is required')
        parameters = np.array([[b.amplitude, b.frequency, b.fdot, 0.,
                                self.phase_convention.gw_phase(b.orbital_phase),
                                np.arccos(b.cosi), b.polarization, b.longitude,
                                np.arcsin(b.sin_latitude)] for b in binaries])
        self.backend.run_wave(*parameters.T, N=self.waveform_bins,
                              dt=self.cadence, T=self.grid.duration)
        channels = np.stack([np.asarray(self.backend.A), np.asarray(self.backend.E)], axis=1)
        return align_waveforms(channels, self.backend.start_inds, self.grid)


class EllcAdapter:
    """Circular spherical eclipse model; beaming/heating/light-time effects disabled.

    Evaluate in orbital cycles (period=1) with our own changing-period mapping.
    Exposure integration occurs in seconds BEFORE the phase transformation.
    """
    def __init__(self, order=12, lc_function=None):
        if lc_function is None:
            import ellc
            lc_function = ellc.lc
        self.lc, self.order = lc_function, order

    def __call__(self, binary, geometry, times, exposures):
        def flux_at_time(sample_times):
            phase = shared_orbital_phase(sample_times, binary.frequency, binary.fdot,
                                         binary.orbital_phase)
            flux = self.lc(t_obs=np.remainder(phase, 1.0), period=1.0, t_zero=0.,
                           radius_1=geometry.radius_1, radius_2=geometry.radius_2,
                           sbratio=geometry.surface_brightness_ratio,
                           incl=np.degrees(np.arccos(abs(binary.cosi))),
                           q=1.0, a=None, f_c=0., f_s=0.,
                           shape_1='sphere', shape_2='sphere',
                           ld_1='lin', ld_2='lin', ldc_1=[geometry.limb_1], ldc_2=[geometry.limb_2],
                           bfac_1=0., bfac_2=0., heat_1=0., heat_2=0.)
            return np.asarray(flux)*geometry.flux_scale
        return exposure_average(times, exposures, flux_at_time, self.order)
