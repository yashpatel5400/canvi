import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"
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
from losses import favi_loss, iwbo_loss, elbo_loss, lebesgue
from generate import generate_data
from utils import transform_parameters
from setup import setup

def loss_choice(loss_name, x, **kwargs):
    if loss_name == 'iwbo':
        return iwbo_loss(x, **kwargs)
    elif loss_name == 'elbo':
        return elbo_loss(x, **kwargs)
    elif loss_name == 'favi':
        return favi_loss(**kwargs)
    else:
        raise ValueError('Specify an appropriate loss name string.')

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:
    # initialize(config_path=".", job_name="test_app")
    # cfg = compose(config_name="config")
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = cfg.training.device

    #torch.cuda.set_device(device)

    dir = cfg.dir
    os.chdir(dir)

    cfg.smc.K = 67
    cfg.training.mb_size = 1

    (true_theta, 
    true_x, 
    logger_string,
    encoder,
    optimizer,
    kwargs) = setup(cfg)

    loss_name = kwargs['loss']
    losses = []
    mean_lebesgue = []
    std_lebesgue = []
    for j in range(kwargs['epochs']):
        optimizer.zero_grad()
        loss = loss_choice(loss_name, true_x, **kwargs)
        print('Loss iter {} is {}'.format(j, loss.item()))

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        del loss

        # Log efficiency
        calibration_theta, calibration_x = generate_data(20, **kwargs)
        lps = encoder.log_prob(calibration_theta, calibration_x).detach()
        cal_scores = -1*lps.reshape(-1)

        areas = lebesgue(cal_scores, calibration_theta, calibration_x, .05, **kwargs)
        mean_area = np.mean(np.array(areas))
        std_area = np.std(np.array(areas))
        mean_lebesgue.append(mean_area)
        std_lebesgue.append(std_area)


    losses = np.array(losses)
    np.save('./logs/{}loss.npy'.format(logger_string), losses)
    mean_lebesgue = np.array(mean_lebesgue)
    np.save('./logs/{}mean.npy'.format(logger_string), mean_lebesgue)
    std_lebesgue = np.array(std_lebesgue)
    np.save('./logs/{}std.npy'.format(logger_string), std_lebesgue)
    torch.save(encoder.state_dict(), './weights/{}.pth'.format(logger_string))

if __name__ == "__main__":
    main() 

