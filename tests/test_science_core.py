import numpy as np
import pytest
from lisa_emgw.core import (FrequencyGrid,GWData,complex_noise,align_waveforms,ModelCache,
                             shared_orbital_phase,exposure_average,Photometry)
from lisa_emgw.benchmark import make_problem,posterior
from lisa_emgw.metrics import wilson_interval, intervals, waveform_errors
from lisa_emgw.native import PhaseConvention,GBGPUAdapter,Binary,EMGeometry,EllcAdapter
from lisa_emgw.experiment import Coordinates,PhysicalLikelihood,observing_schedule,simulate


def test_noise_quadrature_variance_and_normalization():
    psd=np.full((2,100000),3.)
    df=.01
    n=complex_noise(psd,df,np.random.default_rng(4))
    assert np.var(n.real)==pytest.approx(3/(4*df),rel=.015)
    assert np.var(n.imag)==pytest.approx(3/(4*df),rel=.015)
    assert np.mean(np.abs(n)**2)==pytest.approx(3/(2*df),rel=.015)


def test_likelihood_normalization_changes_with_psd():
    g=FrequencyGrid(1,8,10)
    d=GWData(g,np.zeros((2,8),complex),np.ones((2,8)))
    delta=d.loglike(d.strain,[2,2])-d.loglike(d.strain)
    assert delta==pytest.approx(-16*np.log(2))
    h=np.ones((2,8),complex)
    assert d.loglike(h)-d.loglike(d.strain)==pytest.approx(-.5*d.inner(h,h))


def test_absolute_bin_alignment():
    g=FrequencyGrid(10,4,100)
    waves=np.arange(1,13).reshape(2,2,3)
    out=align_waveforms(waves,[9,12],g)
    np.testing.assert_array_equal(out[0],[[2,3,0,0],[5,6,0,0]])
    np.testing.assert_array_equal(out[1],[[0,0,7,8],[0,0,10,11]])


def test_cache_accepted_and_rejected_updates():
    rng=np.random.default_rng(10)
    waves=rng.normal(size=(2,2,8)).astype(complex)
    cache=ModelCache(waves)
    replacement=np.ones((2,8))
    proposal=cache.proposed_total(0,replacement)
    np.testing.assert_allclose(proposal,replacement+waves[1])
    np.testing.assert_array_equal(cache.total,waves.sum(axis=0))
    cache.accept(0,replacement)
    np.testing.assert_allclose(cache.total,cache.full_total())
    external=cache.total; external[:]=0
    np.testing.assert_allclose(cache.total,replacement+waves[1])


def test_exposure_integration_before_phase_mapping():
    t=np.array([0.,3.]); dt=np.array([2.,4.])
    np.testing.assert_allclose(exposure_average(t,dt,lambda x:x*x),t*t+dt*dt/12)
    phase=shared_orbital_phase(t,.1,.002,.25)
    np.testing.assert_allclose(phase,.25+.05*t+.0005*t*t)


def test_orbital_half_cycle_gw_symmetry():
    convention=PhaseConvention(.3,-1,'synthetic interface test only')
    assert convention.gw_phase(.1)==pytest.approx(convention.gw_phase(.6))
    with pytest.raises(ValueError): PhaseConvention(0,1,'')


def test_gaussian_information_transfer_limits():
    for overlap in [0.,.85]:
        data,t,truth,m,eta,sigma=make_problem(overlap)
        _,cg,_,_=posterior(data,t,m,eta,sigma,'G')
        _,cj,_,_=posterior(data,t,m,eta,sigma,'J')
        assert cj[0,0]<cg[0,0]
        if overlap==0: assert cj[1,1]==pytest.approx(cg[1,1])
        else: assert cj[1,1]<cg[1,1]
        # Direct Schur complement agrees with marginalized target precision.
        h=np.linalg.inv(cg)
        h[0,0]+=1/(sigma*sigma+1.)
        np.testing.assert_allclose(cj[:2,:2],np.linalg.inv(h))


def test_sequential_joint_gaussian_equivalence():
    data,t,truth,m,eta,sigma=make_problem(.6)
    mean_g,cov_g,_,_=posterior(data,t,m,eta,sigma,'G')
    mean_j,cov_j,_,_=posterior(data,t,m,eta,sigma,'J')
    precision=np.linalg.inv(cov_g)
    info=precision@mean_g
    precision[0,0]+=1/(sigma*sigma+1.)
    info[0]+=m/(sigma*sigma+1.)
    np.testing.assert_allclose(np.linalg.solve(precision,info),mean_j[:2])


def test_metrics_reject_nonfinite_and_report_counts():
    with pytest.raises(ValueError): intervals([[1],[np.nan]],[1])
    lo,hi=wilson_interval(90,100)
    assert lo < .9 < hi
    with pytest.raises(ValueError): wilson_interval(1,0)


def test_native_adapter_contract_without_native_code():
    class Fake:
        def run_wave(self,*args,**kwargs):
            self.args=args
            self.A=np.array([[1,2,3]]);self.E=np.array([[4,5,6]])
            self.start_inds=np.array([9])
    backend=Fake(); grid=FrequencyGrid(10,4,1000)
    model=GBGPUAdapter(grid,PhaseConvention(.2,1,'mock test'),cadence_seconds=1,backend=backend)
    binary=Binary(1e-22,.01,-1e-17,.1,.3,.4,.5,.2)
    waves=model([binary])
    np.testing.assert_array_equal(waves[0],[[2,3,0,0],[5,6,0,0]])
    assert backend.args[4][0]==pytest.approx(4*np.pi*.1+.2)
    assert backend.args[2][0]<0 # signed fdot supported


def test_photometry_jitter_normalization():
    p=Photometry(np.arange(3.),np.ones(3),np.ones(3),np.ones(3))
    assert p.loglike(np.ones(3),1)-p.loglike(np.ones(3),0)==pytest.approx(-1.5*np.log(2))


def test_ellc_adapter_uses_exposure_nodes_and_correct_arguments():
    calls=[]
    def fake(**kwargs):
        calls.append(kwargs)
        return 1+.1*np.cos(2*np.pi*kwargs['t_obs'])
    binary=Binary(1e-22,.0026,1e-17,.1,.03,.4,.5,.2)
    geom=EMGeometry(.05,.04,.6,.3,.4)
    model=EllcAdapter(order=8,lc_function=fake)
    y=model(binary,geom,np.array([0.,100.]),np.array([10.,10.]))
    assert y.shape==(2,)
    assert calls[0]['t_obs'].shape==(16,)
    assert calls[0]['f_c']==0 and calls[0]['bfac_1']==0 and calls[0]['a'] is None
    assert calls[0]['period']==1


def test_arms_share_models_and_only_joint_uses_em():
    grid=FrequencyGrid(10,4,1000)
    gw=GWData(grid,np.zeros((2,4),complex),np.ones((2,4)))
    em=Photometry(np.arange(3.),np.ones(3),np.ones(3)*1.1,np.ones(3)*.1)
    geom=EMGeometry(.05,.04,.6,.3,.4,jitter=.01)
    coords=Coordinates(.01,1000,(.5,.2),geom)
    x=np.array([-50,0,0,.1,.3,.4, -50,1,0,.2,.2,.4,.6,.3])
    calls=[]
    def wg(sources): return np.stack([np.ones((2,4),complex)*b.amplitude for b in sources])
    def ef(*args):calls.append(1);return np.ones(3)
    g=PhysicalLikelihood(gw,em,wg,ef,coords,'G')
    f=PhysicalLikelihood(gw,em,wg,ef,coords,'F')
    j=PhysicalLikelihood(gw,em,wg,ef,coords,'J')
    jx=np.r_[x,.05,.04,np.log(.6),.3,.4,0,np.log(.01)]
    assert f(x)[0]-g(x)[0]==pytest.approx(em.loglike(np.ones(3),.01))
    assert f(x)[0]==pytest.approx(j(jx)[0])
    assert len(calls)==3


def test_schedule_and_simulation_reproducibility():
    times,exp=observing_schedule([0,1],hours=.1)
    assert len(times)==48 and times[0]==5
    grid=FrequencyGrid(10,4,1000)
    geom=EMGeometry(.05,.04,.6,.3,.4)
    binary=Binary(1e-22,.01,0,.1,.3,.4,.5,.2)
    def wg(sources):return np.ones((len(sources),2,4),complex)
    def ef(*args):return np.ones(len(times))
    args=(grid,np.ones((2,4)),[binary,binary],geom,times,exp,.001,wg,ef,40,41)
    a,b=simulate(*args),simulate(*args)
    np.testing.assert_array_equal(a[0].strain,b[0].strain)
    np.testing.assert_array_equal(a[1].flux,b[1].flux)
    assert not np.array_equal(a[1].flux,np.ones(len(times)))
