import torch
from utils import prior_t_sample, resample, transform_thetas
import torch.distributions as D
import numpy as np

def generate_data(n_samples=100, return_theta=False, **kwargs):
    generator = kwargs['generator']
    noise = kwargs['noise']
    scale = kwargs['scale']
    multiplicative_noise = kwargs['multiplicative_noise']
    thetas = prior_t_sample(n_samples, **kwargs)

    tthetas = transform_thetas(thetas, **kwargs)
    mag = 10.5*torch.ones(n_samples).reshape(-1,1)
    all_params = torch.cat([mag, tthetas], dim=-1)
    outwave, outflux = generator.seds(tt=all_params, zred=torch.zeros(n_samples))

    good_indices = np.isfinite(outflux).all(1)
    bad_indices = ~good_indices
    
    to_resample_wave = outwave[good_indices]
    to_resample_flux = outflux[good_indices]
    params_to_return = thetas[good_indices]
    try:
        new_fluxes = resample(to_resample_wave, to_resample_flux, **kwargs)
    except:
        raise ValueError

    #Resample and smooth
    new_fluxes = resample(outwave, outflux, **kwargs)
    new_fluxes = new_fluxes * scale
    new_fluxes = 1000*new_fluxes/new_fluxes.sum(1).reshape(-1,1)
    new_fluxes = torch.tensor(new_fluxes)

    if not return_theta:
        return D.Normal(new_fluxes, multiplicative_noise*torch.abs(new_fluxes)).sample()
    else:
        return params_to_return, D.Normal(new_fluxes, multiplicative_noise*torch.abs(new_fluxes)).sample()


    # if not return_theta:
    #     return D.Normal(torch.tensor(new_fluxes * scale), multiplicative_noise*torch.abs(torch.tensor(new_fluxes * scale))).sample()
    # else:
    #     return thetas, D.Normal(torch.tensor(new_fluxes * scale), multiplicative_noise*torch.abs(torch.tensor(new_fluxes * scale))).sample()
    
def generate_data_deterministic(n_samples=100, return_theta=True, **kwargs):
    generator = kwargs['generator']
    noise = kwargs['noise']
    scale = kwargs['scale']
    multiplicative_noise = kwargs['multiplicative_noise']
    thetas = prior_t_sample(n_samples, **kwargs)

    tthetas = transform_thetas(thetas, **kwargs)
    mag = 10.5*torch.ones(n_samples).reshape(-1,1)
    all_params = torch.cat([mag, tthetas], dim=-1)
    outwave, outflux = generator.seds(tt=all_params, zred=torch.zeros(n_samples))

    good_indices = np.isfinite(outflux).all(1)
    bad_indices = ~good_indices
    
    to_resample_wave = outwave[good_indices]
    to_resample_flux = outflux[good_indices]
    params_to_return = thetas[good_indices]
    try:
        new_fluxes = resample(to_resample_wave, to_resample_flux, **kwargs)
    except:
        raise ValueError


    #Resample and smooth
    new_fluxes = resample(outwave, outflux, **kwargs)
    new_fluxes = new_fluxes * scale
    new_fluxes = 1000*new_fluxes/new_fluxes.sum(1).reshape(-1,1)
    new_fluxes = torch.tensor(new_fluxes)

    if not return_theta:
        return new_fluxes
    else:
        return params_to_return, new_fluxes
    

def generate_data_emulator(n_samples=100, return_theta=True, **kwargs):
    emulator = kwargs['emulator']
    multiplicative_noise = kwargs['multiplicative_noise']
    device = kwargs['device']

    thetas = prior_t_sample(n_samples, **kwargs).to(device)
    means = emulator(thetas).clamp(min=0.)

    samples = D.Normal(means, multiplicative_noise*torch.abs(means)+1e-8).sample()

    if not return_theta:
        return samples
    else:
        return thetas, samples