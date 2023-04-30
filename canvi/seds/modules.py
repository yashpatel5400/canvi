import torch.nn as nn
import torch
import torch.distributions as D
import numpy as np
import zeus
import h5py
from provabgs.models import NMF
from provabgs.flux_calib import no_flux_factor
from utils import resample
from speclite import filters as specFilter
from provabgs.infer import _MCMC, desiMCMC
from torch.utils.data import Dataset

class TransformedUniform(nn.Module):
    '''
    A Uniform Random variable defined on the real line.
    Transformed by sigmoid to reside between low and high.
    '''
    def __init__(self, low=0., high=1.):
        super().__init__()
        self.low = low
        self.high = high
        self.length = high-low
        self.instance = D.Uniform(low=self.low, high=self.high)
        self.jitter = 1e-8

    def transform(self, value: torch.Tensor):
        tt = torch.sigmoid(value)*self.length+self.low
        clamped = tt.clamp(min=self.low+self.jitter, max=self.high-self.jitter)
        return clamped
    
    def inv_transform(self, tval: torch.Tensor):
        assert (self.low <= tval).all(), "Input is outside of the support."
        assert (self.high >= tval).all(), "Input is outside of the support."
        to_invert = (tval-self.low)/self.length
        return torch.logit(to_invert)
    
    def log_prob(self, value: torch.Tensor):
        tval = self.transform(value)
        return self.instance.log_prob(tval)
    
    def sample(self, shape):
        tsamples = self.instance.sample(shape)
        return self.inv_transform(tsamples)

class TransformedFlatDirichlet(nn.Module): 
    '''
    A transformed dirichlet distribution, where sampling occurs on 
    and n-1 dimensional space. We follow the warped manifold tranformation
    from Betancourt (2013) https://arxiv.org/pdf/1010.3436.pdf.

    The transform from an n-1 collection of Unif(0,1) r.v's to a
    n-dimensional Dirichlet is described in the above. We further
    use a logit transform on the (0,1) space to operate on an unconstrained space.
    '''
    def __init__(self, dim=4):
        super().__init__()
        self.concentration = torch.ones(dim)
        self.ndim = len(self.concentration)
        self.ndim_sampling = self.ndim-1

    def transform(self, tt): 
        ''' 
        tt is a (...,n-1) shaped tensor.
        Return a (..., n) shaped tensor. 
        '''
        tt = torch.sigmoid(tt)
        tt_d = torch.empty(tt.shape[:-1]+(self.ndim,)) 
    
        tt_d[...,0] = 1. - tt[...,0]
        for i in range(1,self.ndim_sampling): 
            tt_d[...,i] = torch.prod(tt[...,:i], axis=-1) * (1. - tt[...,i]) 
        tt_d[...,-1] = torch.prod(tt, axis=-1) 
        return tt_d 

    def inv_transform(self, tt_d): 
        ''' reverse the warped manifold transformation
        i.e. go from n dimensions to n-1.

        Afterward, go from n-1 observations on (0,1) to real numbers
        by logit transformation.
        '''
        assert tt_d.shape[-1] == self.ndim 
        tt = torch.empty(tt_d.shape[:-1]+(self.ndim_sampling,)) 

        tt[...,0] = 1. - tt_d[...,0]
        for i in range(1,self.ndim_sampling): 
            tt[...,i] = 1. - (tt_d[...,i]/torch.prod(tt[...,:i], axis=-1))
        return torch.logit(tt)

    def log_prob(self, theta):
        '''
        Assume that theta is on the sampling space. Unsure if valid for anything
        other than a flat dirichlet distribution.
        '''
        ttheta = self.transform(theta)
        assert ttheta.shape[-1] == self.ndim, "Provided observations reside in wrong dimensional space"
        return D.Dirichlet(self.concentration).log_prob(ttheta)

    def sample_actual(self, shape): 
        return D.Dirichlet(self.concentration).sample(shape)
    
    def sample(self, shape):
        transformed = self.sample_actual(shape)
        return self.inv_transform(transformed)
    
class OurDesiMCMC(desiMCMC): 
    def __init__(self, model=None, flux_calib=None, prior=None, kwargs=None): 
        if model is None: # default Model class object 
            self.model = NMF(burst=False, emulator=True)
        else: 
            self.model = model

        if flux_calib is None: # default FluxCalib function  
            self.flux_calib = no_flux_factor
        else: 
            self.flux_calib = flux_calib

        self.prior = prior 
        assert 'sed' in self.prior.labels, 'please label which priors are for the SED'

        self._filters = None
        self.kwargs = kwargs

    
    def run(self, wave_obs=None, flux_obs=None, flux_ivar_obs=None,
            resolution=None, photo_obs=None, photo_ivar_obs=None, zred=None,
            vdisp=150., tage=None, d_lum=None, mask=None, bands=None, 
            sampler='emcee', nwalkers=100, niter=1000, burnin=100, maxiter=200000,
            opt_maxiter=100, theta_start=None, writeout=None, overwrite=False, debug=False,
            progress=True, pool=None, **kwargs): 
        
        lnpost_args, lnpost_kwargs = self._lnPost_args_kwargs(
                wave_obs=wave_obs, flux_obs=flux_obs,
                flux_ivar_obs=flux_ivar_obs, resolution=resolution, 
                photo_obs=photo_obs, photo_ivar_obs=photo_ivar_obs, zred=zred,
                vdisp=vdisp, mask=mask, bands=bands, tage=tage, d_lum=d_lum)

        self._lnpost_args = lnpost_args
        self._lnpost_kwargs = lnpost_kwargs
        
        # run MCMC 
        if sampler == 'emcee': 
            mcmc_sampler = self._emcee
        elif sampler == 'zeus': 
            mcmc_sampler = self._zeus

        output = mcmc_sampler( 
                lnpost_args, 
                lnpost_kwargs, 
                nwalkers=nwalkers,
                burnin=burnin, 
                niter=niter, 
                maxiter=maxiter,
                opt_maxiter=opt_maxiter, 
                theta_start=theta_start, 
                writeout=writeout, 
                overwrite=overwrite, 
                progress=progress, 
                pool=pool,
                debug=debug) 
        return output  

    def lnLike(self, tt, wave_obs, flux_obs, flux_ivar_obs, photo_obs,
            photo_ivar_obs, zred, vdisp, tage=None, d_lum=None,
            resolution=None, mask=None, filters=None, obs_data_type=None,
            debug=False):
        ''' calculated the log likelihood. 
        '''
        # separate SED parameters from Flux Calibration parameters
        tt_sed, tt_fcalib = self.prior.separate_theta(tt, 
                labels=['sed', 'flux_calib'])
        
        # calculate SED model(theta) 
        _sed = self.model.sed(tt_sed, zred, vdisp=vdisp, wavelength=wave_obs,
                resolution=resolution, filters=filters, tage=tage, d_lum=d_lum)
        if 'photo' in obs_data_type: _, _flux, photo = _sed
        else: _, _flux = _sed

        _chi2_spec, _chi2_photo = 0., 0.
        if 'spec' in obs_data_type: 
            flux = _flux
            flux = resample(wave_obs.reshape(1,-1), flux.reshape(1,-1), **self.kwargs)
            flux = flux * self.kwargs['scale']
            flux = flux.reshape(-1)
            mask = np.array([False for x in range(len(flux))])
            ###----------- END  -----------###

            # data - model(theta) with masking 
            
            dflux = (flux[~mask] - flux_obs[~mask].numpy()) 
            if debug: print(dflux)

            # calculate chi-squared for spectra
            _chi2_spec = np.sum(dflux**2 * flux_ivar_obs[~mask]) 

            if debug: print('desiMCMC.lnLike: Spectroscopic Chi2 = %f' % _chi2_spec)

        if 'photo' in obs_data_type: 
            # data - model(theta) for photometry  
            dphoto = (photo - photo_obs) 
            # calculate chi-squared for photometry 
            _chi2_photo = np.sum(dphoto**2 * photo_ivar_obs) 
            if debug: print('desiMCMC.lnLike: Photometric Chi2 = %f' % _chi2_photo)

        if debug: print('desiMCMC.lnLike: total Chi2 = %f' % (_chi2_spec + _chi2_photo))

        chi2 = _chi2_spec + _chi2_photo
        return -0.5 * chi2

class FAVIDataset(Dataset):
    def __init__(self, thetas, xs):
        self.thetas = thetas
        self.xs = xs

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        return self.xs[idx], self.thetas[idx]
    
class PROVABGSEmulator(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.emulator = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,dim_out)
        )
    def forward(self, params):
        return self.emulator(params)
    


