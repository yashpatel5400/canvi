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
from decimal import Decimal
import json
from cde.mdn import MixtureDensityNetwork
from cde.nsf import build_nsf, EmbeddingNet
import torch.nn as nn
from scipy.signal import savgol_filter
from generate import generate_data_emulator
# from utils import posterior, normalizing_integral, transform_parameters
from setup import setup
from losses import lebesgue, get_log_prob

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


def volume_trial(encoder, kwargs, total_trials=10, trial_sims=1):
    calibration_theta, calibration_x = generate_data_emulator(5000, return_theta=True, **kwargs)
    lps = get_log_prob(encoder, calibration_x, calibration_theta, kwargs['device']).detach()
    cal_scores = 1 / lps.reshape(-1).cpu().exp().numpy()

    mc_effs = []
    alpha = .05
    for _ in range(total_trials):
        fresh_theta, fresh_x = generate_data_emulator(trial_sims, return_theta=True, **kwargs)
        mc_eff = lebesgue(cal_scores, fresh_theta, fresh_x, alpha, **kwargs)
        mc_effs.append(mc_eff)
    print(f"mc_effs: {Decimal(np.mean(mc_effs)):.4E} ({Decimal(np.std(mc_effs)):.4E})")
    return pd.DataFrame(np.vstack([mc_effs]).T, columns=["Estimate"])

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

    # cfg.smc.skip = True
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

    # plot_eff()
    final_effs = {}

    training_to_path = {
        "iwbo": "./weights/iwbo,mdn,0.0001,1000,noise=0.1,mult=0.1,smooth=False.pth",
        "elbo": "./weights/elbo,mdn,0.0001,1000,noise=0.1,mult=0.1,smooth=False.pth",
        "favi": "./weights/favi,mdn,0.0001,1000,noise=0.1,mult=0.1,smooth=False.pth",
    }

    for training_loss in training_to_path:
        print(f"Computing: {training_loss}")
        encoder.load_state_dict(torch.load(training_to_path[training_loss]))
        encoder = encoder.to(device)
        encoder.eval()
        kwargs['encoder'] = encoder
        
        df = volume_trial(encoder, kwargs)
        with open(f"./figs/final_eff_{training_loss}.tex",'w') as tf:
            tf.write(df.to_latex())

if __name__ == "__main__":
    main() 

