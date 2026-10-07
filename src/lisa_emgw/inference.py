"""Eryn runner with independent seeds and explicit burn/production separation."""
import numpy as np
from contextlib import contextmanager
from threading import RLock

_rng_lock = RLock()

@contextmanager
def _legacy_random_seed(seed):
    # Eryn 1.2.6 temperature swaps use global np.random instead of sampler RNG.
    # Serialize this runner in-process; use separate processes for parallel fits.
    with _rng_lock:
        previous = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(previous)



def sample_eryn(loglike, prior, initial, steps, burn, seed, periodic=None):
    """initial shape: (temperatures, walkers, ndim). Returns full cold chain.

    Caller supplies proper Eryn-compatible prior and prior-valid starting points.
    No convergence claim follows from a successful run. Backend files are handled
    by the CLI output layer instead of silently appending to existing chains.
    """
    from eryn.ensemble import EnsembleSampler
    from eryn.state import State
    x = np.array(initial, dtype=float, copy=True)
    if x.ndim != 3 or not np.all(np.isfinite(x)): raise ValueError('Invalid initial ensemble')
    ntemps, nwalkers, ndim = x.shape
    if nwalkers < 2*ndim or steps < 1 or burn < 0:
        raise ValueError('Require >=2*ndim walkers, positive steps and nonnegative burn')
    lp = prior.logpdf(x.reshape(-1, ndim))
    if not np.all(np.isfinite(lp)): raise ValueError('Initial points outside prior support')
    ll = np.asarray(loglike(x.reshape(-1, ndim)))
    if ll.shape != (ntemps*nwalkers,) or not np.all(np.isfinite(ll)):
        raise ValueError('Every initial point needs a finite likelihood')
    kwargs = {} if ntemps == 1 else {'tempering_kwargs': {'ntemps': ntemps, 'stop_adaptation': burn}}
    with _legacy_random_seed(seed):
        sampler = EnsembleSampler(nwalkers, ndim, loglike, prior, vectorize=True,
                                  periodic=None if periodic is None else {'model_0': periodic}, **kwargs)
        sampler.random_state = np.random.RandomState(seed).get_state()
        state = State(x[:, :, None, :], log_like=ll.reshape(ntemps, nwalkers))
        sampler.run_mcmc(state, steps, burn=burn, progress=False)
        chain = sampler.get_chain()['model_0'][:, 0, :, 0, :]
        if chain.shape != (steps, nwalkers, ndim):
            raise RuntimeError(f'Unexpected cold-chain shape {chain.shape}; verify Eryn version semantics')
        return chain, {'acceptance_fraction': np.asarray(sampler.acceptance_fraction).tolist(),
                       'steps': steps, 'burn': burn, 'walkers': nwalkers,
                       'temperatures': ntemps, 'seed': seed,
                       'convergence': 'not_established_by_this_runner'}
