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
        optimizer,
        kwargs
    ) = setup(cfg)

    device = 'cpu'
    kwargs['device'] = device
    kwargs['emulator'] = kwargs['emulator'].to(device)
    test_theta, test_x = generate_data_emulator(cfg.plots.n_test_points, return_theta=True, **kwargs)
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
        logger_string = '{},{},{},{},noise={},mult={},smooth={}'.format(loss, 'flow', cfg.plots.lr, kwargs['K'], kwargs['noise'], kwargs['multiplicative_noise'], kwargs['smooth'])
        encoder.load_state_dict(torch.load('weights/weights_{}'.format(logger_string)))
        encoder = encoder.to(device)
        encoder.eval()
        kwargs['encoder'] = encoder

        for j in range(10):
            print('Trial number {}'.format(j))

            try:
                test_theta, test_x = generate_data_emulator(cfg.plots.n_test_points, return_theta=True, **kwargs)
                assessment_hpr = assess_calibration_hpr(test_theta, test_x, logger_string, mdn=False, flow=True, alphas=alphas, **kwargs)
                calib_results_hpr[str(loss)].append(assessment_hpr)
                # assessment = assess_calibration(test_theta, test_x, logger_string, mdn, flow, alphas=alphas, **kwargs)
                # calib_results[str(loss)].append(assessment)
            except:
                continue

    means_hpr = {}
    stds_hpr = {}

    for key, value in calib_results_hpr.items():
        means_hpr[key] = torch.mean(torch.stack(value), 0).cpu()
        stds_hpr[key] = torch.std(torch.stack(value), 0).cpu()

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


    # means = {}
    # stds = {}

    # for key, value in calib_results.items():
    #     means[key] = torch.mean(torch.stack(value), 0).cpu()
    #     stds[key] = torch.std(torch.stack(value), 0).cpu()

    # param1 = {}
    # param2 = {}
    # param3 = {}
    # param4 = {}
    # param5 = {}
    # param6 = {}
    # param7 = {}
    # param8 = {}
    # param9 = {}
    # param10 = {}
    # for key, value in means.items():
    #     param1[key] = ['{0:.4f}'.format(num) for num in list(value[:,0].numpy())]
    #     param2[key] = ['{0:.4f}'.format(num) for num in list(value[:,1].numpy())]
    #     param3[key] = ['{0:.4f}'.format(num) for num in list(value[:,2].numpy())]
    #     param4[key] = ['{0:.4f}'.format(num) for num in list(value[:,3].numpy())]
    #     param5[key] = ['{0:.4f}'.format(num) for num in list(value[:,4].numpy())]
    #     param6[key] = ['{0:.4f}'.format(num) for num in list(value[:,5].numpy())]
    #     param7[key] = ['{0:.4f}'.format(num) for num in list(value[:,6].numpy())]
    #     param8[key] = ['{0:.4f}'.format(num) for num in list(value[:,7].numpy())]
    #     param9[key] = ['{0:.4f}'.format(num) for num in list(value[:,8].numpy())]
    #     param10[key] = ['{0:.4f}'.format(num) for num in list(value[:,9].numpy())]

    # for key, value in stds.items():
    #     std_strs_1 = [' ({0:.4f})'.format(num) for num in list(value[:,0].numpy())]
    #     std_strs_2 = [' ({0:.4f})'.format(num) for num in list(value[:,1].numpy())]
    #     std_strs_3 = [' ({0:.4f})'.format(num) for num in list(value[:,2].numpy())]
    #     std_strs_4 = [' ({0:.4f})'.format(num) for num in list(value[:,3].numpy())]
    #     std_strs_5 = [' ({0:.4f})'.format(num) for num in list(value[:,4].numpy())]
    #     std_strs_6 = [' ({0:.4f})'.format(num) for num in list(value[:,5].numpy())]
    #     std_strs_7 = [' ({0:.4f})'.format(num) for num in list(value[:,6].numpy())]
    #     std_strs_8 = [' ({0:.4f})'.format(num) for num in list(value[:,7].numpy())]
    #     std_strs_9 = [' ({0:.4f})'.format(num) for num in list(value[:,8].numpy())]
    #     std_strs_10 = [' ({0:.4f})'.format(num) for num in list(value[:,9].numpy())]
    #     param1[key] = map(add, param1[key], std_strs_1)
    #     param2[key] = map(add, param2[key], std_strs_2)
    #     param3[key] = map(add, param3[key], std_strs_3)
    #     param4[key] = map(add, param4[key], std_strs_4)
    #     param5[key] = map(add, param5[key], std_strs_5)
    #     param6[key] = map(add, param6[key], std_strs_6)
    #     param7[key] = map(add, param7[key], std_strs_7)
    #     param8[key] = map(add, param8[key], std_strs_8)
    #     param9[key] = map(add, param9[key], std_strs_9)
    #     param10[key] = map(add, param10[key], std_strs_10)

    # results1 = pd.DataFrame(param1)
    # results2 = pd.DataFrame(param2)
    # results3 = pd.DataFrame(param3)
    # results4 = pd.DataFrame(param4)
    # results5 = pd.DataFrame(param5)
    # results6 = pd.DataFrame(param6)
    # results7 = pd.DataFrame(param7)
    # results8 = pd.DataFrame(param8)
    # results9 = pd.DataFrame(param9)
    # results10 = pd.DataFrame(param10)
    # results1 = results1.set_axis(alphas.detach().numpy(), axis='index')
    # results2 = results2.set_axis(alphas.detach().numpy(), axis='index')
    # results3 = results3.set_axis(alphas.detach().numpy(), axis='index')
    # results4 = results4.set_axis(alphas.detach().numpy(), axis='index')
    # results5 = results5.set_axis(alphas.detach().numpy(), axis='index')
    # results6 = results6.set_axis(alphas.detach().numpy(), axis='index')
    # results7 = results7.set_axis(alphas.detach().numpy(), axis='index')
    # results8 = results8.set_axis(alphas.detach().numpy(), axis='index')
    # results9 = results9.set_axis(alphas.detach().numpy(), axis='index')
    # results10 = results10.set_axis(alphas.detach().numpy(), axis='index')
    # results1.rename(mapper=mapper, inplace=True, axis=1)
    # results2.rename(mapper=mapper, inplace=True, axis=1)
    # results3.rename(mapper=mapper, inplace=True, axis=1)
    # results4.rename(mapper=mapper, inplace=True, axis=1)
    # results5.rename(mapper=mapper, inplace=True, axis=1)
    # results6.rename(mapper=mapper, inplace=True, axis=1)
    # results7.rename(mapper=mapper, inplace=True, axis=1)
    # results8.rename(mapper=mapper, inplace=True, axis=1)
    # results9.rename(mapper=mapper, inplace=True, axis=1)
    # results10.rename(mapper=mapper, inplace=True, axis=1)

    # with open('./figs/lr={},K={},theta1.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
    #     tf.write(results1.to_latex())
    # with open('./figs/lr={},K={},theta2.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
    #     tf.write(results2.to_latex())
    # with open('./figs/lr={},K={},theta3.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
    #     tf.write(results3.to_latex())
    # with open('./figs/lr={},K={},theta4.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
    #     tf.write(results4.to_latex())
    # with open('./figs/lr={},K={},theta5.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
    #     tf.write(results5.to_latex())
    # with open('./figs/lr={},K={},theta6.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
    #     tf.write(results6.to_latex())
    # with open('./figs/lr={},K={},theta7.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
    #     tf.write(results7.to_latex())
    # with open('./figs/lr={},K={},theta8.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
    #     tf.write(results8.to_latex())
    # with open('./figs/lr={},K={},theta9.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
    #     tf.write(results9.to_latex())
    # with open('./figs/lr={},K={},theta10.tex'.format(cfg.plots.lr, kwargs['K']),'w') as tf:
    #     tf.write(results10.to_latex())


if __name__ == "__main__":
    main()