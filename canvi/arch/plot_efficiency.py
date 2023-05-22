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
from os.path import exists
import torch.distributions as D
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import random
import json
from cde.mdn import MixtureDensityNetwork
from cde.nsf import build_nsf, EmbeddingNet
import torch.nn as nn
from scipy.signal import savgol_filter
from generate import generate_data_favi
from utils import posterior, normalizing_integral, transform_parameters
from setup import setup
from losses import lebesgue

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 28})

def plot_eff():
    
    fig, ax = plt.subplots(figsize=(20,20))
    iwbo = np.load('./logs/iwbo,0.0001,10mean.npy')
    fil_iwbo = savgol_filter(iwbo, 10, 3)
    iwbo_err = np.load('./logs/iwbo,0.0001,10std.npy')
    lower_iwbo = iwbo-iwbo_err
    upper_iwbo = iwbo+iwbo_err
    ax.plot(np.arange(1,len(iwbo)+1)*500, fil_iwbo, c='r', label='IWBO')
    # ax[0].fill_between(np.arange(len(iwbo)), lower_iwbo, upper_iwbo)
    ax.set_xlim(0, len(iwbo)*500)
    ax.set_ylim(0,2)

    elbo = np.load('./logs/elbo,0.0001,10mean.npy')
    elbo_err = np.load('./logs/elbo,0.0001,10std.npy')
    fil_elbo = savgol_filter(elbo,10,3)
    lower_elbo = elbo-elbo_err
    upper_iwbo = elbo+elbo_err
    ax.plot(np.arange(1,len(iwbo)+1)*500, fil_elbo, c='b', label='ELBO')
    # ax[0].fill_between(np.arange(len(iwbo)), lower_iwbo, upper_iwbo)

    favi = np.load('./logs/favi,0.0001,10mean.npy')
    favi_err = np.load('./logs/favi,0.0001,10std.npy')
    fil_favi = savgol_filter(favi,10,3)
    lower_favi = favi-favi_err
    upper_favi = favi+favi_err
    ax.plot(np.arange(1,len(iwbo)+1)*500, fil_favi, c='g', label='FAVI')
    ax.set_ylabel('Efficiency')
    ax.set_xlabel('Training Step')
    # ax[0].fill_between(np.arange(len(iwbo)), lower_iwbo, upper_iwbo)
    plt.legend(loc="lower left")
    plt.savefig("./figs/eff.png")
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

    device = 'cpu'
    kwargs['device'] = device

    plot_eff()
    final_effs = {}

    # IWBO
    encoder.load_state_dict(torch.load('./weights/iwbo,0.0001,10.pth'))
    encoder = encoder.to(device)
    calibration_theta, calibration_x = generate_data_favi(1000, **kwargs)
    lps = encoder.log_prob(calibration_theta, calibration_x).detach()
    cal_scores = -1*lps.reshape(-1)
    iwbo_effs = []
    alphas = [.05]
    for _ in range(1):
        fresh_theta, fresh_x = generate_data_favi(1000, **kwargs)
        this_eff = [lebesgue(cal_scores, fresh_theta, fresh_x, alpha, **kwargs) for alpha in alphas]
        iwbo_effs.append(this_eff)
    iwbo_effs = np.stack(iwbo_effs).mean(-1)
    iwbo_df = pd.DataFrame(iwbo_effs, columns=alphas)

    # ELBO
    encoder.load_state_dict(torch.load('./weights/elbo,0.0001,10.pth'))
    encoder = encoder.to(device)
    calibration_theta, calibration_x = generate_data_favi(1000, **kwargs)
    lps = encoder.log_prob(calibration_theta, calibration_x).detach()
    cal_scores = -1*lps.reshape(-1)
    elbo_effs = []
    for _ in range(1):
        fresh_theta, fresh_x = generate_data_favi(1000, **kwargs)
        this_eff = [lebesgue(cal_scores, fresh_theta, fresh_x, alpha, **kwargs) for alpha in alphas]
        elbo_effs.append(this_eff)
    elbo_effs = np.stack(elbo_effs).mean(-1)
    elbo_df = pd.DataFrame(elbo_effs, columns=alphas)

    # FAVI
    encoder.load_state_dict(torch.load('./weights/favi,0.0001,10.pth'))
    encoder = encoder.to(device)
    calibration_theta, calibration_x = generate_data_favi(1000, **kwargs)
    lps = encoder.log_prob(calibration_theta, calibration_x).detach()
    cal_scores = -1*lps.reshape(-1)
    favi_effs = []
    for _ in range(1):
        fresh_theta, fresh_x = generate_data_favi(1000, **kwargs)
        this_eff = [lebesgue(cal_scores, fresh_theta, fresh_x, alpha, **kwargs) for alpha in alphas]
        favi_effs.append(this_eff)
    favi_effs = np.stack(favi_effs).mean(-1)
    favi_df = pd.DataFrame(favi_effs, columns=alphas)

    with open('./figs/final_eff_elbo.tex','w') as tf:
        tf.write(elbo_df.to_latex())
    with open('./figs/final_eff_iwbo.tex','w') as tf:
        tf.write(iwbo_df.to_latex())
    with open('./figs/final_eff_favi.tex','w') as tf:
        tf.write(favi_df.to_latex())

if __name__ == "__main__":
    main() 

