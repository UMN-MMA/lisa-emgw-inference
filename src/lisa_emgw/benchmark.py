"""Analytic linear-Gaussian infrastructure test; NOT a LISA/ellc forecast.

EM is deliberately a linear constraint on target amplitude plus a nuisance
parameter. No physical light-curve interpretation should be assigned to it.
"""
import numpy as np
from .core import FrequencyGrid, GWData, complex_noise
from .metrics import intervals, waveform_errors


def make_problem(overlap=.8, seed=42, noise=True):
    if not np.isfinite(overlap) or abs(overlap) >= 1: raise ValueError('Require |overlap| < 1')
    grid = FrequencyGrid(100, 64, 1000.)
    psd = np.ones((2, grid.count))
    zero = GWData(grid, np.zeros_like(psd, dtype=complex), psd)
    basis_rng = np.random.default_rng(17)
    u = basis_rng.normal(size=psd.shape)+1j*basis_rng.normal(size=psd.shape)
    u /= np.sqrt(zero.inner(u,u))
    v = basis_rng.normal(size=psd.shape)+1j*basis_rng.normal(size=psd.shape)
    v -= zero.inner(u,v)*u
    v /= np.sqrt(zero.inner(v,v))
    templates = np.stack([u, overlap*u+np.sqrt(1-overlap**2)*v])
    truth = np.array([8., 5.])
    rng = np.random.default_rng(seed)
    n = complex_noise(psd, grid.df, rng) if noise else np.zeros_like(u)
    data = GWData(grid, np.sum(truth[:,None,None]*templates, axis=0)+n, psd)
    eta, em_sigma = .25, .2
    em_rng = np.random.default_rng(seed+1)
    measurement = truth[0]+eta+(em_rng.normal()*em_sigma if noise else 0.)
    return data, templates, truth, measurement, eta, em_sigma


def posterior(data, templates, measurement, eta, sigma, arm):
    if arm not in ('T','G','F','J'): raise ValueError('Unknown arm')
    count = 1 if arm=='T' else 2
    ndim = count+(arm=='J')
    precision = np.zeros((ndim, ndim))
    information = np.zeros(ndim)
    for i in range(count):
        information[i] = data.inner(templates[i], data.strain)
        for j in range(count): precision[i,j] = data.inner(templates[i],templates[j])
    if arm in ('F','J'):
        em_design = np.zeros(ndim); em_design[0] = 1.
        if arm=='J': em_design[-1] = 1.
        value = measurement-eta if arm=='F' else measurement
        precision += np.outer(em_design,em_design)/sigma**2
        information += em_design*value/sigma**2
    prior_std = np.full(ndim,10.)
    if arm=='J': prior_std[-1] = 1.
    covariance = np.linalg.inv(precision+np.diag(1/prior_std**2))
    mean = covariance@information
    def loglike(x):
        x = np.atleast_2d(x)
        return -.5*np.einsum('bi,ij,bj->b',x,precision,x)+x@information
    return mean, covariance, prior_std, loglike


def run_benchmark(overlap=.8, seed=42, draws=20000, sampler='analytic', steps=1000, burn=500):
    if draws < 2: raise ValueError('At least two draws required')
    data, templates, truth, measurement, eta, sigma = make_problem(overlap, seed)
    report = {'kind':'linear_gaussian_validation_not_physical_LISA_result', 'overlap':overlap,
              'seed':seed, 'sampler':sampler, 'noise_convention':'one_sided_PSD', 'arms':{}}
    arrays = {'data':data.strain, 'psd':data.psd, 'templates':templates, 'truth':truth,
              'em_measurement':np.array(measurement)}
    for index, arm in enumerate(('T','G','F','J')):
        mean,cov,std,loglike = posterior(data,templates,measurement,eta,sigma,arm)
        rng = np.random.default_rng(seed+100+index)
        diagnostics = {'draws':'independent_exact_gaussian'}
        if sampler=='analytic': samples = rng.multivariate_normal(mean,cov,size=draws)
        elif sampler=='eryn':
            from scipy.stats import norm
            from eryn.prior import ProbDistContainer
            from .inference import sample_eryn
            prior = ProbDistContainer({i:norm(0,s) for i,s in enumerate(std)})
            initial = rng.normal(size=(1,24,len(mean)))*std
            chain,diagnostics = sample_eryn(loglike,prior,initial,steps,burn,seed+200+index)
            arrays[f'chain_{arm}'] = chain
            samples = chain.reshape(-1,len(mean))
        else: raise ValueError('sampler must be analytic or eryn')
        reference = truth[:1] if arm=='T' else np.r_[truth,eta] if arm=='J' else truth
        summary = intervals(samples,reference)
        count = 1 if arm=='T' else 2
        components,total = waveform_errors(data,samples[:,:count,None,None]*templates[None,:count],
                                           truth[:count,None,None]*templates[:count])
        # T errors describe its target only, not successful recovery of the omitted neighbour.
        summary.update(analytic_mean=mean.tolist(),analytic_covariance=cov.tolist(),
                       mean_error_in_analytic_sigma=((samples.mean(axis=0)-mean)/np.sqrt(np.diag(cov))).tolist(),
                       mean_component_waveform_error=components.mean(axis=0).tolist(),
                       mean_fitted_components_waveform_error=float(total.mean()),diagnostics=diagnostics)
        report['arms'][arm] = summary
        arrays[f'samples_{arm}'] = samples
    report['target_width_ratio_G_over_J'] = report['arms']['G']['width_90'][0]/report['arms']['J']['width_90'][0]
    report['neighbour_width_ratio_G_over_J'] = report['arms']['G']['width_90'][1]/report['arms']['J']['width_90'][1]
    return report,arrays
