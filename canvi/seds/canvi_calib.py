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
from utils import transform_thetas, put_to_right_form
from modules import OurDesiMCMC
from utils import transform_thetas
from setup import setup
from modules import TransformedFlatDirichlet, TransformedUniform, OurDesiMCMC, FAVIDataset
from generate import generate_data_emulator
import torch
from operator import add
from utils import prior_t_sample, resample, transform_thetas
import torch.distributions as D

def assess_calibration(thetas, x, logger_string, mdn=True, flow=False, n_samples=10000, alphas=.05, **kwargs):
    assert not (mdn and flow), "One of mdn or flow flags must be false."
    encoder = kwargs['encoder']
    device = kwargs['device']

    results = torch.zeros(alphas.shape[0], thetas.shape[1]).to(device)
    for j in range(x.shape[0]):
        true_param = thetas[j]
        observation = x[j]
        # Sample from encoder
        if mdn:
            log_pi, mu, sigma = encoder(x[j].to(device))
            mix = D.Categorical(logits=log_pi.view(-1))
            comp = D.Independent(D.Normal(mu.squeeze(0), sigma.squeeze(0)), 1)
            mixture = D.MixtureSameFamily(mix, comp)
            particles = mixture.sample((n_samples,)).clamp(-1., 1.)
        elif flow:
            particles = encoder.sample(num_samples=n_samples, context=observation.view(1,-1).to(device))
        particles = particles.reshape(n_samples, -1)

        for j in range(alphas.shape[0]):
            alpha = alphas[j]
            q = torch.tensor([alpha/2, 1-alpha/2]).to(device)
            quantiles = torch.quantile(particles, q, dim=0)
            success = ((true_param > quantiles[0]) & (true_param < quantiles[1])).long()
            results[j] += success

    return results/x.shape[0]

def assess_calibration_canvi(cal_scores, thetas, x, logger_string, n_samples=10000, alphas=.05, **kwargs):
    encoder = kwargs['encoder']
    device = kwargs['device']

    results = torch.zeros(alphas.shape[0], thetas.shape[1])
    hey = [0., 0., 0., 0.]
    for j in range(x.shape[0]):
        true_param = thetas[j]
        observation = x[j]
        # Sample from encoder
        particles = prior_t_sample(100000, **kwargs)
        lps = encoder.log_prob(particles.to(device), x[j].view(1,-1).repeat(particles.shape[0],1).to(device)).detach()
        lps = lps.exp().cpu()
        scores = 1/lps

        # particles = encoder.sample(num_samples=n_samples, context=observation.view(1,-1).to(device))
        # particles = particles.reshape(n_samples, -1)

        for kk in range(alphas.shape[0]):
            alpha = alphas[kk]
            q = torch.tensor([1-alpha])
            quantiles = torch.quantile(cal_scores, q, dim=0)
            hpr = scores < quantiles[0]
            samples_keep = particles[hpr]
            mins = torch.min(samples_keep, dim=0)[0]
            maxs = torch.max(samples_keep, dim=0)[0]
            

            success = ((true_param.cpu() > mins) & (true_param.cpu() < maxs)).all()
            if success:
                hey[kk] += 1.

            #results[j] += success

    return results/x.shape[0]

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

    (
        thetas,
        seds,
        epochs,
        device,
        mb_size,
        encoder,
        mdn,
        flow,
        logger_string,
        writer,
        optimizer,
        loss_fcn,
        kwargs
    ) = setup(cfg)

    # Calibration scores
    calibration_theta, calibration_x = generate_data_emulator(10_000, return_theta=True, **kwargs)
    cal_scores = []
    for calibration_theta_pt, calibration_x_pt in zip(calibration_theta, calibration_x):
        log_prob = encoder.log_prob(calibration_theta_pt.view(1,-1).to(device), calibration_x_pt.view(1,-1).to(device)).detach()
        prob = log_prob.cpu().exp().numpy()
        cal_scores.append(1 / prob)
    cal_scores = np.array(cal_scores)
    cal_scores = torch.tensor(cal_scores).reshape(-1)

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
        logger_string = '{},{},{},{},noise={},mult={},smooth={}'.format(loss, 'flow', cfg.plots.lr, kwargs['K'], kwargs['noise'], kwargs['multiplicative_noise'], kwargs['smooth'])
        encoder.load_state_dict(torch.load('weights/weights_{}'.format(logger_string)))
        encoder.eval()
        kwargs['encoder'] = encoder

        for j in range(10):
            print('Trial number {}'.format(j))

            try:
                test_theta, test_x = generate_data_emulator(cfg.plots.n_test_points, return_theta=True, **kwargs)
                assessment = assess_calibration(test_theta, test_x, logger_string, mdn, flow, alphas=alphas, **kwargs)
                calib_results[str(loss)].append(assessment)
            except:
                continue

    means = {}
    stds = {}

    for key, value in calib_results.items():
        means[key] = torch.mean(torch.stack(value), 0).cpu()
        stds[key] = torch.std(torch.stack(value), 0).cpu()

    param1 = {}
    param2 = {}
    for key, value in means.items():
        param1[key] = ['{0:.4f}'.format(num) for num in list(value[:,0].numpy())]
        param2[key] = ['{0:.4f}'.format(num) for num in list(value[:,1].numpy())]

    for key, value in stds.items():
        std_strs_1 = [' ({0:.4f})'.format(num) for num in list(value[:,0].numpy())]
        std_strs_2 = [' ({0:.4f})'.format(num) for num in list(value[:,1].numpy())]
        param1[key] = map(add, param1[key], std_strs_1)
        param2[key] = map(add, param2[key], std_strs_2)

    results1 = pd.DataFrame(param1)
    results2 = pd.DataFrame(param2)
    results1 = results1.set_axis(alphas.detach().numpy(), axis='index')
    results2 = results2.set_axis(alphas.detach().numpy(), axis='index')
    results1.rename(mapper=mapper, inplace=True, axis=1)
    results2.rename(mapper=mapper, inplace=True, axis=1)

    with open('./figs/lr={},K={},theta1.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
        tf.write(results1.to_latex())
    with open('./figs/lr={},K={},theta2.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
        tf.write(results2.to_latex())

if __name__ == "__main__":
    main()