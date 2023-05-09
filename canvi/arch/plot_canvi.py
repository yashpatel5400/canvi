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


def plot_before_after_canvi(cal_scores, j, thetas, x, encoder, n_samples=10000, alpha=.05, **kwargs):
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30,15))
    device = kwargs['device']
    theta1vals = torch.arange(-1., 1., .01)
    theta2vals = torch.arange(0., 1., .01)
    data = x[j]
    X, Y = torch.meshgrid(theta1vals, theta2vals)
    Z = torch.empty(X.shape)

    # Get quantile
    particles, lps = encoder.sample_and_log_prob(num_samples=n_samples, context=data.view(1,-1).to(device))
    scores = -1*lps.reshape(-1)
    q = torch.tensor([1-alpha]).to(device)
    quantiles = torch.quantile(scores, q, dim=0)

    eval_pts = torch.cartesian_prod(theta1vals, theta2vals)
    lps = encoder.log_prob(eval_pts.to(device),data.view(1,-1).repeat(eval_pts.shape[0], 1).to(device))
    scores = -1*lps
    in_region = (scores < quantiles[0]).long()
    in_region = in_region.reshape(X.shape).cpu().numpy()

    ax[0].pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), in_region)
    ax[0].plot(thetas.detach().cpu()[j][0],thetas.detach().cpu()[j][1], 'ro', markersize=12)
    percent = int(100*(1-alpha))
    ax[0].set_title('{}% HPR ({}), Observation {}'.format(percent, kwargs['loss'].upper(), j))

    # Get quantile
    quantiles = torch.quantile(cal_scores, q, dim=0)
    in_region = (scores < quantiles[0]).long()
    in_region = in_region.reshape(X.shape).cpu().numpy()

    ax[1].pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), in_region)
    ax[1].plot(thetas.detach().cpu()[j][0],thetas.detach().cpu()[j][1], 'ro', markersize=12)
    ax[1].set_title('Corrected {}% HPR ({}), Observation {}'.format(percent, kwargs['loss'].upper(), j))
    plt.tight_layout()
    plt.savefig('./figs/{}_{}_{}canvi.png'.format(j, kwargs['loss'], alpha))
    plt.clf()




# def plot_hpr_calib(cal_scores, j, thetas, x, logger_string, mdn=True, flow=False, n_samples=10000, alpha=.05, **kwargs):
#     assert not (mdn and flow), "One of mdn or flow flags must be false."
#     encoder = kwargs['encoder']
#     device = kwargs['device']
#     theta1vals = torch.arange(-1., 1., .01)
#     theta2vals = torch.arange(0., 1., .01)
#     data = x[j]
#     X, Y = torch.meshgrid(theta1vals, theta2vals)
#     Z = torch.empty(X.shape)

#     # Get quantile
#     q = torch.tensor([1-alpha])
#     quantiles = torch.quantile(cal_scores, q, dim=0)

#     eval_pts = torch.cartesian_prod(theta1vals, theta2vals)
#     lps = encoder.log_prob(eval_pts.to(device),data.view(1,-1).repeat(eval_pts.shape[0], 1).to(device))
#     scores = -1*lps
#     in_region = (scores < quantiles[0]).long()
#     in_region = in_region.reshape(X.shape).cpu().numpy()

#     fig, ax = plt.subplots(figsize=(10,10))
#     ax.pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), in_region)
#     ax.set_title('Approximate Posterior Flow')

# def plot_hpr(j, thetas, x, logger_string, mdn=True, flow=False, n_samples=10000, alpha=.05, **kwargs):
#     assert not (mdn and flow), "One of mdn or flow flags must be false."
#     encoder = kwargs['encoder']
#     device = kwargs['device']
#     theta1vals = torch.arange(-1., 1., .01)
#     theta2vals = torch.arange(0., 1., .01)
#     data = x[j]
#     X, Y = torch.meshgrid(theta1vals, theta2vals)
#     Z = torch.empty(X.shape)

#     # Get quantile
#     particles, lps = encoder.sample_and_log_prob(num_samples=n_samples, context=data.view(1,-1).to(device))
#     scores = 1/(lps.exp())
#     scores = scores.reshape(-1)
#     q = torch.tensor([1-alpha]).to(device)
#     quantiles = torch.quantile(scores, q, dim=0)


#     eval_pts = torch.cartesian_prod(theta1vals, theta2vals)
#     lps = encoder.log_prob(eval_pts.to(device),data.view(1,-1).repeat(eval_pts.shape[0], 1).to(device))
#     scores = 1/(lps.exp())
#     in_region = (scores < quantiles[0]).long()
#     in_region = in_region.reshape(X.shape).cpu().numpy()

#     fig, ax = plt.subplots(figsize=(10,10))
#     ax.pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), in_region)
#     ax.set_title('Approximate Posterior Flow')

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

    # Calibration scores
    calibration_theta, calibration_x = generate_data(100000, **kwargs)
    lps = encoder.log_prob(calibration_theta, calibration_x).detach()
    cal_scores = -1*lps.reshape(-1)

    # Plotting code
    for j in cfg.plots.points_to_plot:
        for alpha in cfg.plots.alphas:
            plot_before_after_canvi(cal_scores, j, true_theta, true_x, encoder, alpha=alpha, **kwargs)


if __name__ == "__main__":
    main() 
