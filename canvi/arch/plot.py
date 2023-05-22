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

def plot(index, true_theta, true_x, encoder, **kwargs):
    device = kwargs['device']
    my_t_priors = kwargs['my_t_priors']
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30,15))

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

    
    #ax.contour(X, Y, Z, levels=10)
    ax[0].pcolormesh(X, Y, Z)
    ax[0].plot(true_theta.detach().cpu()[index][0],true_theta.detach().cpu()[index][1], 'ro', markersize=12)
    ax[0].set_title('Exact Posterior - Observation {}'.format(index))
    
    # Flow posterior
    vals1normal = torch.arange(-1.+.01, 1., .01)
    vals2normal = torch.arange(0.+.01, 1., .01)
    eval_pts_normal = torch.cartesian_prod(vals1normal, vals2normal)
    eval_pts_uncon = torch.empty(eval_pts_normal.shape)
    eval_pts_uncon[:,0] = my_t_priors[0].inv_transform(eval_pts_normal[:,0])
    eval_pts_uncon[:,1] = my_t_priors[1].inv_transform(eval_pts_normal[:,1])
    lps = encoder.log_prob(eval_pts_uncon.to(device), true_x[index].view(1,-1).repeat(eval_pts_normal.shape[0],1).to(device)).detach()
    X, Y = torch.meshgrid(vals1normal, vals2normal)
    Z = lps.view(X.shape)
    ax[1].pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().exp().numpy())
    ax[1].plot(true_theta.detach().cpu()[index][0], true_theta.detach().cpu()[index][1], 'ro', markersize=12)
    ax[1].set_title('Approximate Posterior Flow ({}) - Observation {}'.format(kwargs['loss'].upper(), index))
    plt.tight_layout()
    plt.savefig('./figs/{}_{}'.format(index, kwargs['loss']))

    return


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:
    # initialize(config_path=".", job_name="test_app")
    # cfg = compose(config_name="config")
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
    kwargs) = setup(cfg)

    kwargs.pop('encoder')

    encoder.load_state_dict(torch.load('./weights/{}.pth'.format(logger_string)))
    encoder.eval()
    # device = kwargs['device']

    # Plotting code
    for j in cfg.plots.points_to_plot:
        plot(j, true_theta, true_x, encoder, **kwargs)


    # index = 499
    # # Exact posterior
    # theta1vals = np.arange(-1., 1., .01)
    # theta2vals = np.arange(0., 1., .01)
    # data = true_x[index].cpu().numpy()
    # X, Y = np.meshgrid(theta1vals, theta2vals)
    # Z = np.empty(X.shape)

    # normalized = normalizing_integral(x=data, **kwargs)[0]
    # for i in range(X.shape[0]):
    #     for j in range(Y.shape[1]):
    #         Z[i,j] = posterior(X[i,j], Y[i,j], x=data, **kwargs)/normalized

    # plt.rcParams.update({'font.size': 22})
    # fig, ax = plt.subplots(figsize=(30,15))
    # ax.contour(X, Y, Z, levels=10)
    # ax.set_xlabel('$\\theta_1$')
    # ax.set_ylabel('$\\theta_2$')
    # plt.savefig('./figs/theta1={},theta2={}.png'.format(true_theta[index,0].item(), true_theta[index,1].item()))

    # index = 244
    # # Exact posterior
    # theta1vals = np.arange(-1., 1., .01)
    # theta2vals = np.arange(0., 1., .01)
    # data = true_x[index].cpu().numpy()
    # X, Y = np.meshgrid(theta1vals, theta2vals)
    # Z = np.empty(X.shape)

    # normalized = normalizing_integral(x=data, **kwargs)[0]
    # for i in range(X.shape[0]):
    #     for j in range(Y.shape[1]):
    #         Z[i,j] = posterior(X[i,j], Y[i,j], x=data, **kwargs)/normalized

    # fig, ax = plt.subplots(figsize=(30,15))
    # ax.contour(X, Y, Z, levels=10)
    # ax.set_xlabel('$\\theta_1$')
    # ax.set_ylabel('$\\theta_2$')
    # plt.savefig('./figs/theta1={},theta2={}.png'.format(true_theta[index,0].item(), true_theta[index,1].item()))

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

    # # 2d hist
    # device = kwargs['device']
    # log_pi, mu, sigma = encoder(true_x[index].view(1,-1).to(device))
    # mix = D.Categorical(logits=log_pi)
    # comp = D.Independent(D.Normal(mu, sigma), 1)
    # mixture = D.MixtureSameFamily(mix, comp)
    # particles = mixture.sample((10000,))
    # particles = particles.reshape(10000, -1)
    # # ttheta = transform_parameters(particles, **kwargs)
    # ttheta = particles
    # fig, ax = plt.subplots(figsize=(15,15))
    # ax.hist2d(ttheta[:,0].detach().cpu().numpy(), ttheta[:,1].detach().cpu().numpy(), bins=100, range=[[-1.,1.], [0.,1.]])


    # # Assessing calibration
    # encoder = encoder.to(device)
    # test_theta, test_x = generate_data(1000, **kwargs)
    # assess_calibration(test_theta, test_x, n_samples=1000, alpha=.1, **kwargs)

    # assess_calibration(true_theta, true_x, n_samples=1000, alpha=.05, **kwargs)


if __name__ == "__main__":
    main() 

