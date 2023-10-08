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
from utils import prior_t_sample, transform_parameters, transform_parameters_batch, log_prior, log_prior_batch, log_target

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

    logger_string = '{},{},{}'.format(cfg.training.loss, cfg.training.lr, K)
    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.training.lr)
    
    kwargs['encoder'] = encoder
    loss_name = cfg.training.loss
    kwargs['loss'] = loss_name

    return (true_theta, 
            true_x, 
            logger_string,
            encoder,
            optimizer,
            kwargs)

 




