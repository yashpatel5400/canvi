import os
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
import torch
import torch.distributions as D
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import numpy as np 
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage
import time
from scipy.interpolate import CubicSpline
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False
from os.path import exists
import torch.distributions as D
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import random
from cde.mdn import MixtureDensityNetwork
from cde.nsf import build_nsf, EmbeddingNet
import torch.nn as nn
from generate import generate_data
from losses import favi_loss, iwbo_loss, elbo_loss
from scipy.integrate import quad
from modules import TransformedUniform
from utils import prior_t_sample, transform_parameters, transform_parameters_batch

def log_target(particles, context, num_particles, **kwargs):
    T = kwargs['T']
    device = kwargs['device']
    K = num_particles
    params = transform_parameters_batch(particles, **kwargs).to(device)
    theta1 = params[...,0]
    theta2 = params[...,1]
    eyes = torch.eye(T).repeat(K,1,1).to(device) #batch of Qs
    repeated = -1*theta1.reshape(-1,1).repeat(1, T-1)
    subdiags = torch.diag_embed(repeated, -1)
    Qs = eyes + subdiags
    
    targets = context.reshape(-1,1).repeat(K, 1, 1)
    eps_vecs = torch.bmm(Qs, targets)
    eps_vecs = eps_vecs.squeeze(-1)

    running_sum = torch.zeros(1, K).to(device)
    for i in range(1, T):
        const = 1/torch.sqrt(2*math.pi*(.2+theta2*(eps_vecs[:,i-1]**2)))
        const = torch.log(const).to(device)
        fact = -1*(eps_vecs[:,i]**2)/(2*(.2+theta2*eps_vecs[:,i-1]**2))
        running_sum += const
        running_sum += fact
    return running_sum

def log_prior(particles, **kwargs):
    prior = kwargs['prior']
    theta = transform_parameters(particles, **kwargs)
    lps = prior.log_prob(theta) #
    return lps

def log_prior_batch(particles, **kwargs):
    prior = kwargs['prior']
    theta = transform_parameters_batch(particles, **kwargs)
    lps = prior.log_prob(theta) #
    return lps

def setup(cfg):
    # initialize(config_path=".", job_name="test_app")
    # cfg = compose(config_name="config")
    device = cfg.training.device
    prior = D.Independent(D.Uniform(torch.tensor([-1., 0.]), torch.tensor([1., 1.])), 1)
    my_t_priors = [
        TransformedUniform(-1., 1.),
        TransformedUniform(0., 1.),
    ]
    n_obs = cfg.data.n_pts
    K = cfg.smc.K
    T = cfg.data.T

    kwargs = {
        'n_pts': n_obs,
        'device': device,
        'prior': prior,
        'my_t_priors': my_t_priors,
        'K': K,
        'T': T,
        'log_prior': log_prior,
        'log_prior_batch': log_prior_batch,
        'log_target': log_target,
        'my_t_priors': my_t_priors
    }

    true_theta, true_x = generate_data(n_obs, **kwargs)

    epochs = cfg.training.epochs
    device=cfg.training.device
    mb_size = cfg.training.mb_size

    kwargs['mb_size'] = mb_size
    kwargs['device'] = device
    kwargs['epochs'] = epochs

    # EXAMPLE BATCH FOR SHAPES
    theta_dim = prior.sample().shape[-1]
    x_dim = true_x.shape[-1]
    num_obs_flow = K*mb_size
    fake_zs = torch.randn((K*mb_size, theta_dim))
    fake_xs = torch.randn((K*mb_size, x_dim))
    encoder = build_nsf(fake_zs, fake_xs, z_score_x='structured', z_score_y='structured', hidden_features=64, embedding_net=EmbeddingNet(x_dim).float())
    #encoder = MixtureDensityNetwork(dim_in=x_dim, dim_out=theta_dim, n_components=20, hidden_dim=64)

    logger_string = '{},{},{}'.format(cfg.training.loss, cfg.training.lr, K)
    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.training.lr)
    
    kwargs['encoder'] = encoder

    # clip_value = 1e-1
    # for p in encoder.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    # Select loss function
    loss_name = cfg.training.loss
    kwargs['loss'] = loss_name
    if loss_name == 'elbo':
        loss_fcn = lambda: elbo_loss(true_x, **kwargs)
    elif loss_name == 'iwbo':
        loss_fcn = lambda: iwbo_loss(true_x, **kwargs)
    elif loss_name == 'favi':
        loss_fcn = lambda: favi_loss(true_x, **kwargs)
    else:
        raise ValueError('Specify an appropriate loss name string.')

    return (true_theta, 
            true_x, 
            logger_string,
            encoder,
            optimizer,
            kwargs)

 




