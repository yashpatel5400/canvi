import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
import math
import matplotlib.pyplot as plt
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
from utils import posterior, normalizing_integral, transform_parameters
from setup import setup


def assess_calibration(thetas, x, n_samples=1000, alpha=.05, **kwargs):
    device = kwargs['device']
    encoder = kwargs['encoder']

    results = torch.zeros_like(thetas[0])
    for j in range(x.shape[0]):
        true_param = thetas[j]
        observation = x[j]
        particles, log_denoms = encoder.sample_and_log_prob(num_samples=n_samples, context=x[j].view(1,-1).float().to(device))
        particles = particles.reshape(n_samples, -1)
        q = torch.tensor([alpha/2, 1-alpha/2]).to(device)
        quantiles = torch.quantile(particles, q, dim=0)
        success = ((true_param > quantiles[0]) & (true_param < quantiles[1])).long()
        results += success

    return results/x.shape[0]

def assess_calibration(thetas, x, n_samples=1000, alpha=.05, **kwargs):
    device = kwargs['device']
    encoder = kwargs['encoder']

    results = torch.zeros_like(thetas[0])
    for j in range(x.shape[0]):
        true_param = thetas[j]
        observation = x[j]

        log_pi, mu, sigma = encoder(observation.view(1,-1).to(device))
        mix = D.Categorical(logits=log_pi)
        comp = D.Independent(D.Normal(mu, sigma), 1)
        mixture = D.MixtureSameFamily(mix, comp)
        particles = mixture.sample((10000,))
        particles = particles.reshape(10000, -1)

        q = torch.tensor([alpha/2, 1-alpha/2]).to(device)
        quantiles = torch.quantile(particles, q, dim=0)
        success = ((true_param > quantiles[0]) & (true_param < quantiles[1])).long()
        results += success

    return results/x.shape[0]


def plot(index, true_x, encoder, **kwargs):

    # Exact posterior
    theta1vals = np.arange(-1., 1., .01)
    theta2vals = np.arange(0., 1., .01)
    data = true_x[index].cpu().numpy()
    X, Y = np.meshgrid(theta1vals, theta2vals)
    Z = np.empty(X.shape)

    normalized = normalizing_integral(x=data, **kwargs)[0]
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            Z[i,j] = posterior(X[i,j], Y[i,j], x=data, **kwargs)/normalized

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30,15))
    ax.contour(X, Y, Z, levels=10)

    # Hist 2d
    # fig, ax = plt.subplots(figsize=(15,15))
    # ax.hist2d(samples[:,0].detach().numpy(), samples[:,1].detach().numpy(), bins=200, range=[[-1.,1.], [0.,1.]])
    # ax.set_title('SMC Approx. K={}'.format(K))
    
    # Flow posterior


@hydra.main(version_base=None, config_path="../conf", config_name="config5")
def main(cfg : DictConfig) -> None:
    initialize(config_path="../conf", job_name="test_app")
    cfg = compose(config_name="config5")
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    cfg.smc.skip = True
    (true_theta, 
    true_x, 
    logger_string,
    encoder,
    optimizer,
    loss_fcn,
    kwargs) = setup(cfg)

    encoder.load_state_dict(torch.load('./exp5/weights/{}_mdn{}.pth'.format(logger_string, '100000')))
    encoder.eval()
    device = kwargs['device']


    # Plotting code

    index = 499
    # Exact posterior
    theta1vals = np.arange(-1., 1., .01)
    theta2vals = np.arange(0., 1., .01)
    data = true_x[index].cpu().numpy()
    X, Y = np.meshgrid(theta1vals, theta2vals)
    Z = np.empty(X.shape)

    normalized = normalizing_integral(x=data, **kwargs)[0]
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            Z[i,j] = posterior(X[i,j], Y[i,j], x=data, **kwargs)/normalized

    fig, ax = plt.subplots(figsize=(30,15))
    ax.contour(X, Y, Z, levels=10)

    # # 2d hist
    # device = kwargs['device']
    # particles, log_denoms = encoder.sample_and_log_prob(num_samples=10000, context=true_x[index].view(1,-1).to(device))
    # particles = particles.reshape(10000, -1)
    # # ttheta = transform_parameters(particles, **kwargs)
    # ttheta = particles
    # fig, ax = plt.subplots(figsize=(15,15))
    # ax.hist2d(ttheta[:,0].detach().cpu().numpy(), ttheta[:,1].detach().cpu().numpy(), bins=100, range=[[-1.,1.], [0.,1.]])

    # vals1normal = torch.arange(-1., 1., .01)
    # vals2normal = torch.arange(0., 1., .01)
    # eval_pts_normal = torch.cartesian_prod(vals1normal, vals2normal)
    # lps = encoder.log_prob(eval_pts_normal.to(device), true_x[index].view(1,-1).repeat(eval_pts_normal.shape[0],1).to(device)).detach()
    # X, Y = torch.meshgrid(vals1normal, vals2normal)
    # Z = lps.view(X.shape)
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,15))
    # ax.pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().exp().numpy())
    # ax.set_title('Approximate Posterior Flow')

    # 2d hist
    device = kwargs['device']
    log_pi, mu, sigma = encoder(true_x[index].view(1,-1).to(device))
    mix = D.Categorical(logits=log_pi)
    comp = D.Independent(D.Normal(mu, sigma), 1)
    mixture = D.MixtureSameFamily(mix, comp)
    particles = mixture.sample((10000,))
    particles = particles.reshape(10000, -1)
    # ttheta = transform_parameters(particles, **kwargs)
    ttheta = particles
    fig, ax = plt.subplots(figsize=(15,15))
    ax.hist2d(ttheta[:,0].detach().cpu().numpy(), ttheta[:,1].detach().cpu().numpy(), bins=100, range=[[-1.,1.], [0.,1.]])


    # Assessing calibration
    encoder = encoder.to(device)
    test_theta, test_x = generate_data(1000, **kwargs)
    assess_calibration(test_theta, test_x, n_samples=1000, alpha=.1, **kwargs)

    assess_calibration(true_theta, true_x, n_samples=1000, alpha=.05, **kwargs)






if __name__ == "__main__":
    main() 

