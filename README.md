# LISA EM–GW Inference

Research software for controlled joint electromagnetic–gravitational-wave
inference of a known eclipsing binary and neighbouring Galactic binary signals.

The package provides explicit likelihoods, reproducible simulations, a two-source
inference runner, and an analytic benchmark for validating information transfer
between correlated sources.

## Status

The linear-Gaussian benchmark and its Eryn implementation have been tested.
The physical GBGPU–ellc pipeline is **experimental and not yet validated end to
end**. Native runtime compatibility, the instrument PSD, orbital-phase mapping,
prior sensitivity and independent convergence checks remain prerequisites for
scientific use. The software does not implement a full Galactic global fit or
unknown-source-count inference.

## Installation

Python 3.10 or later is required. From this directory, preferably in a virtual
environment:

```bash
python -m pip install -e '.[test]'
```

The core requires NumPy. Optional dependencies can be installed separately:

```bash
python -m pip install -e '.[sampler]'  # Eryn
python -m pip install -e '.[native]'   # Eryn, GBGPU and ellc
```

The native libraries may require platform-specific runtime libraries. A successful
package installation does not guarantee that native waveform evaluation works.

## Quick start

Run the analytic benchmark with a new output directory:

```bash
python -m lisa_emgw benchmark --overlap 0.8 --out runs/analytic
```

Run the same problem with Eryn:

```bash
python -m lisa_emgw benchmark --sampler eryn --steps 1200 --burn 600 --out runs/eryn
```

Both commands save run metadata, posterior summaries and samples. Output
directories must not already exist. The installed `lisa-emgw` command is an
alternative to `python -m lisa_emgw`.

**These benchmarks are numerical validation problems, not LISA forecasts.**
They use linear templates and an artificial target-amplitude constraint with a
Gaussian nuisance parameter. Their gains must not be interpreted as physical
light-curve results.

## Comparison arms

| Arm | Model |
|---|---|
| T | Target-only GW fit, intentionally omitting the neighbour |
| G | Simultaneous target and neighbour GW fit |
| F | Joint fit with EM nuisance parameters fixed to injected values; optimistic control |
| J | Joint fit marginalizing EM nuisance parameters |

G versus J measures the incremental contribution of EM data. All arms use the
same injection/noise seeds and shared GW priors. Fixing nuisance parameters in F
is not a realistic measurement of their uncertainty.

## Experiment preparation

```bash
python scripts/prepare_campaign.py --stage pilot
python scripts/make_native_config.py --configuration pilot-000 --realization 0 --out configs/my-case.json
```

`experiments/campaign.json` defines the design. The pilot contains 12 configurations
and 144 planned fits before independent sampler repeats. Preparation scripts run
no inference. Native configuration generation works directly from the campaign;
a previously generated manifest is not required. Generated files are never
silently overwritten. The larger main grid is a proposed design, not an executed
campaign or a resource commitment.

## Physical inference

Inspect the example `configs/pilot-000.example.json`. Before running, supply:

1. A PSD NPZ containing `frequencies` with shape `(bins,)` and positive `psd` with
   shape `(2,bins)` in A/E order, using one-sided PSD and matching TDI/FFT units.
   Frequencies must match the configured absolute Fourier grid. No generic strain
   sensitivity curve is silently substituted for a channel PSD.
2. A calibrated orbital-conjunction/GBGPU phase convention: offset, sign and a
   reference to the validation. A nonempty reference field alone is not validation.
3. Reviewed finite prior bounds and an independent initial ensemble per arm in
   `.npy` format, shaped `(temperatures, walkers, ndim)`. Require at least twice as
   many walkers as dimensions. T/G/F/J have 6/14/14/21 dimensions respectively.
4. Working native libraries and convergence checks for frequency-window width,
   exposure quadrature and the sampler.

Then use the same configuration across comparison arms:

```bash
python -m lisa_emgw doctor --native
python -m lisa_emgw run-native --config configs/my-case.json --arm G --out runs/native-G
python -m lisa_emgw run-native --config configs/my-case.json --arm J --out runs/native-J
```

Relative input paths resolve against the configuration's directory. The runner
saves data, configuration, scaled injection, cold chain, scalar summaries and
metadata. Check that data hashes match between arms. Set
`diagnostic_waveform_draws` above zero to compute waveform-reconstruction errors
for a posterior subset; this adds native model evaluations.

The physical baseline uses spherical, circular stars without beaming, irradiation
or inner-binary light-travel corrections. Frequency, evolution and inclination
are shared. Exposure integration happens in physical time before orbital-phase
evaluation. Target sky position is fixed identically in all arms. Source count
is known; neighbour summaries require inspection for association ambiguity.

An execution completing successfully does not certify convergence. Keep independent
ensembles, assess missed modes, and distinguish fixed-truth repeated-noise coverage
from prior-predictive calibration. Eryn temperature-swap reproducibility is handled
by saving/seeding/restoring NumPy's legacy RNG; use separate processes for parallel
fits. The native diagnostic command uses POSIX process groups (macOS/Linux).

## Tests

```bash
python -m pytest -q
```

Tests cover noise normalization, frequency alignment, cached updates, exposure
integration, interface contracts, Gaussian posterior identities, seed handling
and experiment preparation. Eryn integration tests skip when Eryn is unavailable.
Mock native interfaces test contracts, not the physical accuracy of GBGPU or ellc.

## Layout

```text
src/lisa_emgw/   Likelihoods, simulation, adapters, metrics and CLI
scripts/        Deterministic experiment/configuration preparation
experiments/    Campaign design
configs/        Editable native-run example
tests/         Unit and sampler integration tests
```

## Dependencies and references

- [Eryn](https://doi.org/10.1093/mnras/stad2939): posterior sampling.
- [GBGPU](https://github.com/lisa-analysis-tools/GBGPU): Galactic-binary response.
- [ellc](https://doi.org/10.1051/0004-6361/201628579): light-curve modelling.

These external libraries are dependencies, not vendored source. Their respective
licenses and citation requirements continue to apply. This repository does not
include manuscript files, external datasets, posterior chains or machine-specific
validation logs.
