import numpy as np
import pytest


def test_eryn_matches_analytic_gaussian_posterior():
    pytest.importorskip('eryn')
    from lisa_emgw.benchmark import run_benchmark
    report,arrays=run_benchmark(sampler='eryn',steps=800,burn=400)
    for arm,summary in report['arms'].items():
        assert np.max(np.abs(summary['mean_error_in_analytic_sigma'])) < .2
        assert arrays[f'chain_{arm}'].shape[:2] == (800,24)
        assert summary['diagnostics']['convergence']=='not_established_by_this_runner'


def test_tempered_runner_preserves_shape_and_seed():
    pytest.importorskip('eryn')
    from scipy.stats import norm
    from eryn.prior import ProbDistContainer
    from lisa_emgw.inference import sample_eryn
    prior=ProbDistContainer({0:norm(0,2),1:norm(0,2)})
    initial=np.random.default_rng(2).normal(size=(2,10,2))
    loglike=lambda x: -.5*np.sum(np.atleast_2d(x)**2,axis=1)
    original=initial.copy()
    previous=np.random.get_state()
    a,_=sample_eryn(loglike,prior,initial,steps=12,burn=5,seed=19)
    after=np.random.get_state()
    np.testing.assert_array_equal(previous[1],after[1])
    assert previous[2:] == after[2:]
    np.testing.assert_array_equal(initial,original)
    b,_=sample_eryn(loglike,prior,initial,steps=12,burn=5,seed=19)
    assert a.shape==(12,10,2)
    np.testing.assert_array_equal(a,b)
