import torch
from scipy import ndimage
from scipy.interpolate import CubicSpline
import numpy as np
import torch.nn as nn


def transform_thetas(thetas, **kwargs):
    sizes = kwargs['sizes']
    my_t_priors = kwargs['my_t_priors']

    new = torch.empty(thetas.shape[0], 11)
    spot = 0
    for j in range(len(sizes)):
        if j == 0:
            new_spot = sizes[j]+spot
            new[:,spot:new_spot+1] = my_t_priors[j].transform(thetas[:,spot:new_spot])
            spot = new_spot+1
        else:
            new_spot = sizes[j]+spot
            new[:,spot:new_spot] = my_t_priors[j].transform(thetas[:,spot-1:new_spot-1])
            spot = new_spot

    return new

def _resample_one(wave, flux, **kwargs):
        """For a single spectrograph, resample onto rest-frame grid.
        wave, flux are 1D ndarrays."""
        obs_grid = kwargs['obs_grid']
        smooth = kwargs['smooth']
        cs = CubicSpline(wave, flux)
        res = cs(obs_grid)
        smooth_param = kwargs['smooth_parameter']
        if smooth:
            smoothed = ndimage.gaussian_filter1d(res, smooth_param)
            return smoothed
        else:
            return res
            

def resample(waves, fluxes, **kwargs):
    obs_grid = kwargs['obs_grid']
    new_fluxes = np.empty((fluxes.shape[0], len(obs_grid)))
    for i in range(len(fluxes)):
        new_fluxes[i] = _resample_one(waves[i], fluxes[i], **kwargs)
    return new_fluxes

def prior_t_sample(n_obs, **kwargs):
    my_t_priors = kwargs['my_t_priors']
    samples = [prior.sample((n_obs,)) for prior in my_t_priors]
    samples = [x.unsqueeze(-1) if len(x.shape) == 1 else x for x in samples]
    #samples[1] = shrink_dirichlet(samples[1])
    return torch.cat(samples, -1)

def prior_t_sample_tup(shape, **kwargs):
    my_t_priors = kwargs['my_t_priors']
    samples = [prior.sample(shape) for prior in my_t_priors]
    samples = [x.unsqueeze(-1) for x in samples]
    #samples[1] = shrink_dirichlet(samples[1])
    return torch.cat(samples, -1)

def log_t_prior(samples, **kwargs):
    '''
    Give samples of shape (n_samples, 11),
    return (n_samples,) shaped array of
    log prior densities.
    '''
    sizes = kwargs['sizes']
    my_t_priors = kwargs['my_t_priors']

    spot = 0
    lp = []
    for j in range(len(sizes)):
        new = sizes[j]+spot
        part = samples[...,spot:new]
        lp.append(my_t_priors[j].log_prob(part).view(-1))
        spot=new
    return torch.stack(lp).sum(0)

def put_to_right_form(theta, obs_space=True, **kwargs):
    my_t_priors = kwargs['my_t_priors']
    sizes = kwargs['sizes']
    sizes_transformed = kwargs['sizes_transformed']
    n_rows = theta.shape[0]
    to_return = torch.empty((n_rows, 11))
    if obs_space:
        to_return[:,0] = theta[:,0]
        to_return[:,4:] = theta[:,5:]
        to_return[:,1:4] = torch.sigmoid(my_t_priors[1].inv_transform(theta[:,1:5]))
    else:
        ttheta = transform_thetas(theta, **kwargs)
        to_return = put_to_right_form(ttheta, obs_space=True, **kwargs)
    return to_return

def gather_weights(**kwargs):
    n_pts = kwargs['n_pts']
    noise = kwargs['noise']
    multiplicative_noise = kwargs['multiplicative_noise']
    smooth_param = kwargs['smooth_parameter']
    smooth = kwargs['smooth']
    print('Gathering SMC from noise level {}, mult noise {}, smoothing={}'.format(noise,multiplicative_noise, smooth)) 
    
    ws = []
    lws = []
    its = []
    for j in range(n_pts):
        weights = torch.load('./exp3/tmp/weights_noise={},mult={},smooth={},smoothparam={}_{}.pt'.format(noise,multiplicative_noise, smooth,smooth_param, j))
        log_weights = torch.load('./exp3/tmp/log_weights_noise={},mult={},smooth={},smoothparam={}_{}.pt'.format(noise,multiplicative_noise, smooth, smooth_param, j))[:,-1]
        items = torch.load('./exp3/tmp/items_noise={},mult={},smooth={},smoothparam={}_{}.pt'.format(noise,multiplicative_noise, smooth, smooth_param, j))[:,-1,...]

        ws.append(weights)
        lws.append(log_weights)
        its.append(items)
    return torch.stack(ws), torch.stack(lws), torch.stack(its)
    





