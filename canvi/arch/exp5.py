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
from losses import favi_loss, iwbo_loss, elbo_loss
from generate import generate_data
from utils import transform_parameters
from setup import setup

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:
    initialize(config_path=".", job_name="test_app")
    cfg = compose(config_name="config")
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = cfg.training.device

    #torch.cuda.set_device(device)

    dir = cfg.dir
    os.chdir(dir)

    (true_theta, 
    true_x, 
    logger_string,
    encoder,
    optimizer,
    loss_fcn,
    kwargs) = setup(cfg)

    for j in range(100000):
        optimizer.zero_grad()
        loss = loss_fcn()
        print('Loss iter {} is {}'.format(j, loss))

        if loss.item() > 10:
            continue

        loss.backward()
        optimizer.step()
        del loss


    torch.save(encoder.state_dict(), '..weights/{}.pth'.format(logger_string))

if __name__ == "__main__":
    main() 

