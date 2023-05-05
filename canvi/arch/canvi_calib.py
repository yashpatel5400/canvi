import os
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=1
import sys
sys.path.append("../")
import numpy as np 
from provabgs import util as UT
from provabgs import infer as Infer
from provabgs import models as Models
from provabgs import flux_calib as FluxCalib
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import exists
import torch.distributions as D
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.kde import gaussian_kde
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import random
import pandas as pd
import numpy as np
from setup import setup
from generate import generate_data
import torch
from operator import add
import torch.distributions as D

def assess_calibration_canvi(cal_scores, thetas, x, logger_string, n_samples=10000, alphas=.05, **kwargs):
    encoder = kwargs['encoder']

    results = torch.zeros(alphas.shape[0])

    for kk in range(alphas.shape[0]):
        alpha = alphas[kk]
        q = torch.tensor([1-alpha])
        quantiles = torch.quantile(cal_scores, q, dim=0)

        # Sample from encoder
        lps = encoder.log_prob(thetas, x).detach()
        probs = lps.exp().cpu()
        scores = 1/probs

        success = (scores <= quantiles[0]).long().sum()
        results[kk] = success/x.shape[0]

    return results


def plot_hpr_calib(cal_scores, j, thetas, x, logger_string, mdn=True, flow=False, n_samples=10000, alpha=.05, **kwargs):
    assert not (mdn and flow), "One of mdn or flow flags must be false."
    encoder = kwargs['encoder']
    device = kwargs['device']
    theta1vals = torch.arange(-1., 1., .01)
    theta2vals = torch.arange(0., 1., .01)
    data = x[j]
    X, Y = torch.meshgrid(theta1vals, theta2vals)
    Z = torch.empty(X.shape)

    # Get quantile
    q = torch.tensor([1-alpha])
    quantiles = torch.quantile(cal_scores, q, dim=0)

    eval_pts = torch.cartesian_prod(theta1vals, theta2vals)
    lps = encoder.log_prob(eval_pts.to(device),data.view(1,-1).repeat(eval_pts.shape[0], 1).to(device))
    scores = 1/(lps.exp())
    in_region = (scores < quantiles[0]).long()
    in_region = in_region.reshape(X.shape).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), in_region)
    ax.set_title('Approximate Posterior Flow')

def plot_hpr(j, thetas, x, logger_string, mdn=True, flow=False, n_samples=10000, alpha=.05, **kwargs):
    assert not (mdn and flow), "One of mdn or flow flags must be false."
    encoder = kwargs['encoder']
    device = kwargs['device']
    theta1vals = torch.arange(-1., 1., .01)
    theta2vals = torch.arange(0., 1., .01)
    data = x[j]
    X, Y = torch.meshgrid(theta1vals, theta2vals)
    Z = torch.empty(X.shape)

    # Get quantile
    particles, lps = encoder.sample_and_log_prob(num_samples=n_samples, context=data.view(1,-1).to(device))
    scores = 1/(lps.exp())
    scores = scores.reshape(-1)
    q = torch.tensor([1-alpha]).to(device)
    quantiles = torch.quantile(scores, q, dim=0)


    eval_pts = torch.cartesian_prod(theta1vals, theta2vals)
    lps = encoder.log_prob(eval_pts.to(device),data.view(1,-1).repeat(eval_pts.shape[0], 1).to(device))
    scores = 1/(lps.exp())
    in_region = (scores < quantiles[0]).long()
    in_region = in_region.reshape(X.shape).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), in_region)
    ax.set_title('Approximate Posterior Flow')

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:
    initialize(config_path=".", job_name="test_app")
    cfg = compose(config_name="config")
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

    kwargs.pop('encoder')
    kwargs.pop('loss')
    
    mapper = {
    'elbo': 'ELBO',
    'iwbo': 'IWBO',
    'favi': 'FAVI',
    }

    names = list(map(lambda name: mapper[name], cfg.plots.losses))
    alphas = torch.tensor(cfg.plots.alphas)
    calib_results = {}
    for loss in cfg.plots.losses:
        calib_results[str(loss)] = []

    for loss in cfg.plots.losses:
        print('Working on {}'.format(loss))
        logger_string = '{},{},{}.pth'.format(loss, cfg.plots.lr, kwargs['K'])
        encoder.load_state_dict(torch.load('weights/{}'.format(logger_string)))
        encoder.eval()
        kwargs['encoder'] = encoder

        # Calibration scores
        calibration_theta, calibration_x = generate_data(100000, **kwargs)
        lps = encoder.log_prob(calibration_theta, calibration_x).detach()
        probs = lps.cpu().exp().numpy()
        cal_scores = 1/probs
        cal_scores = torch.tensor(cal_scores).reshape(-1)

        for j in range(10):
            print('Trial number {}'.format(j))
            try:
                test_theta, test_x = generate_data(1000, **kwargs)
                assessment = assess_calibration_canvi(cal_scores, test_theta, test_x, logger_string, alphas=alphas, **kwargs)
                calib_results[str(loss)].append(assessment)
            except:
                continue

    means = {}
    stds = {}

    for key, value in calib_results.items():
        means[key] = torch.mean(torch.stack(value), 0).cpu().numpy()
        stds[key] = torch.std(torch.stack(value), 0).cpu().numpy()

    final = {}
    for key, value in means.items():
        final[key] = ['{0:.4f}'.format(num) for num in list(value)]

    for key, value in stds.items():
        std_strs = [' ({0:.4f})'.format(num) for num in list(value)]
        final[key] = map(add, final[key], std_strs)

    final = pd.DataFrame(final)
    final = final.set_axis(alphas.detach().numpy(), axis='index')
    final.rename(mapper=mapper, inplace=True, axis=1)

    with open('./figs/lr={},K={},all_canvi.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
        tf.write(final.to_latex())

if __name__ == "__main__":
    main()