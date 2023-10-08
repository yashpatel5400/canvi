import os
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=1
import sys
sys.path.append("../")
import numpy as np 
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
import torch
from operator import add
from generate import generate_data_favi
import torch.distributions as D

def assess_calibration_dimensionwise(thetas, x, logger_string, mdn=True, flow=False, n_samples=1000, alphas=[.05], **kwargs):
    assert not (mdn and flow), "One of mdn or flow flags must be false."
    encoder = kwargs['encoder']
    device = kwargs['device']

    results = torch.zeros(alphas.shape[0], thetas.shape[1]).to(device)

    particles = encoder.sample(n_samples, x.to(device))
    #particles = particles.reshape(n_samples, -1)

    for j in range(alphas.shape[0]):
        alpha = alphas[j]
        q = torch.tensor([alpha/2, 1-alpha/2]).to(device)
        quantiles = torch.quantile(particles, q, dim=1)
        success = ((thetas > quantiles[0]) & (thetas < quantiles[1])).long().float()
        results[j] += success.mean(0)

    return results

def assess_calibration_hpr(thetas, x, logger_string, mdn=True, flow=False, n_samples=1000, alphas=[.05], **kwargs):
    encoder = kwargs['encoder']
    device = kwargs['device']

    results = torch.zeros(alphas.shape[0]).to(device)
    particles, lps = encoder.sample_and_log_prob(n_samples, x.to(device))
    scores = -1*lps

    scores_at_truth = -1*encoder.log_prob(thetas, x.to(device))

    for j in range(alphas.shape[0]):
        alpha = alphas[j]
        q = torch.tensor([1-alpha]).to(device)
        quantiles = torch.quantile(scores, q, dim=1).reshape(-1)
        success = ((scores_at_truth < quantiles)).long().float()
        results[j] += success.mean(0)

    return results


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

    cfg.training.device = 'cpu'

    cfg.smc.skip = True
    (true_theta, 
    true_x, 
    logger_string,
    encoder,
    optimizer,
    kwargs) = setup(cfg)

    #test_theta, test_x = generate_data(cfg.plots.n_test_points, **kwargs)
    kwargs.pop('encoder')
    kwargs.pop('loss')

    mapper = {
    'elbo': 'ELBO',
    'iwbo': 'IWBO',
    'favi': 'FAVI',
    }

    names = list(map(lambda name: mapper[name], cfg.plots.losses))
    alphas = torch.tensor(cfg.plots.alphas)
    calib_results_dimensionwise = {}
    calib_results_hpr = {}
    for loss in cfg.plots.losses:
        calib_results_dimensionwise[str(loss)] = []
        calib_results_hpr[str(loss)] = []

    for loss in cfg.plots.losses:
        print('Working on {}'.format(loss))
        logger_string = '{},{},{}.pth'.format(loss, cfg.plots.lr, kwargs['K'])
        encoder.load_state_dict(torch.load('weights/{}'.format(logger_string)))
        encoder.eval()
        kwargs['encoder'] = encoder

        for j in range(10):
            print('Trial number {}'.format(j))
            try:
                test_theta, test_x = generate_data_favi(cfg.plots.n_test_points, **kwargs)
                assessment_dim = assess_calibration_dimensionwise(test_theta, test_x, logger_string, mdn=False, flow=True, alphas=alphas, **kwargs)
                calib_results_dimensionwise[str(loss)].append(assessment_dim)
                assessment_hpr = assess_calibration_hpr(test_theta, test_x, logger_string, mdn=False, flow=True, alphas=alphas, **kwargs)
                calib_results_hpr[str(loss)].append(assessment_hpr)
            except:
                continue

    means = {}
    stds = {}
    means_hpr = {}
    stds_hpr = {}

    for key, value in calib_results_dimensionwise.items():
        means[key] = torch.mean(torch.stack(value), 0).cpu()
        stds[key] = torch.std(torch.stack(value), 0).cpu()
    for key, value in calib_results_hpr.items():
        means_hpr[key] = torch.mean(torch.stack(value), 0).cpu()
        stds_hpr[key] = torch.std(torch.stack(value), 0).cpu()

    # Dimensionwise Formatting + Results
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
    
    results1 = results1.set_axis((1-alphas).detach().numpy(), axis='index')
    results2 = results2.set_axis((1-alphas).detach().numpy(), axis='index')
    
    results1.rename(mapper=mapper, inplace=True, axis=1)
    results2.rename(mapper=mapper, inplace=True, axis=1)
    

    with open('./figs/lr={},K={},theta1.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
        tf.write(results1.to_latex())
    with open('./figs/lr={},K={},theta2.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
        tf.write(results2.to_latex())

    # HPR Formatting + Results
    final = {}
    for key, value in means_hpr.items():
        final[key] = ['{0:.4f}'.format(num) for num in list(value.numpy())]
    
    for key, value in stds_hpr.items():
        std_strs = [' ({0:.4f})'.format(num) for num in list(value.numpy())]
        final[key] = map(add, final[key], std_strs)
    final = pd.DataFrame(final)
    final = final.set_axis((1-alphas).detach().numpy(), axis='index')
    final.rename(mapper=mapper, inplace=True, axis=1)
    with open('./figs/lr={},K={},hpr.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
        tf.write(final.to_latex())

    

if __name__ == "__main__":
    main()