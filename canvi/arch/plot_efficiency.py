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
import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import random
from cde.mdn import MixtureDensityNetwork
from cde.nsf import build_nsf, EmbeddingNet
import torch.nn as nn
from scipy.signal import savgol_filter
from generate import generate_data
from utils import posterior, normalizing_integral, transform_parameters
from setup import setup

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 28})

def plot_eff():
    
    fig, ax = plt.subplots(figsize=(20,20))
    iwbo = np.load('./logs/iwbo,0.0001,10mean.npy')
    fil_iwbo = savgol_filter(iwbo, 20, 3)
    iwbo_err = np.load('./logs/iwbo,0.0001,10std.npy')
    lower_iwbo = iwbo-iwbo_err
    upper_iwbo = iwbo+iwbo_err
    ax.plot(np.arange(len(iwbo)), fil_iwbo, c='r', label='IWBO')
    # ax[0].fill_between(np.arange(len(iwbo)), lower_iwbo, upper_iwbo)
    ax.set_xlim(0,10000)
    ax.set_ylim(0,2)

    elbo = np.load('./logs/elbo,0.0001,10mean.npy')
    elbo_err = np.load('./logs/elbo,0.0001,10std.npy')
    fil_elbo = savgol_filter(elbo,20,3)
    lower_elbo = elbo-elbo_err
    upper_iwbo = elbo+elbo_err
    ax.plot(np.arange(len(elbo)), fil_elbo, c='b', label='ELBO')
    # ax[0].fill_between(np.arange(len(iwbo)), lower_iwbo, upper_iwbo)

    favi = np.load('./logs/favi,0.0001,10mean.npy')
    favi_err = np.load('./logs/favi,0.0001,10std.npy')
    fil_favi = savgol_filter(favi,20,3)
    lower_favi = favi-favi_err
    upper_favi = favi+favi_err
    ax.plot(np.arange(len(favi)), fil_favi, c='g', label='FAVI')
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

    plot_eff()

if __name__ == "__main__":
    main() 

