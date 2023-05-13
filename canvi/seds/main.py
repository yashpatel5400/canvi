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
from losses import iwbo_loss, elbo_loss, favi_loss
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
import random
from setup import setup

def loss_choice(loss_name, x, **kwargs):
    if loss_name == 'iwbo':
        return iwbo_loss(x, mdn=True, flow=False, **kwargs)
    elif loss_name == 'elbo':
        return elbo_loss(x, mdn=True, flow=False, **kwargs)
    elif loss_name == 'favi':
        return favi_loss(mdn=True, flow=False, **kwargs)
    else:
        raise ValueError('Specify an appropriate loss name string.')


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
    cfg.smc.only=False
    #cfg.training.device = 'cpu'

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

    loss_name = kwargs['loss']
    losses = []
    writer = SummaryWriter('./logs/{}'.format(logger_string))
    for j in range(epochs):
        if j % 1000 == 0:
            print("On iteration {}".format(j))
        optimizer.zero_grad()
        loss = loss_choice(loss_name, seds, **kwargs)

        print('Loss iter {} is {}'.format(j, loss))
        if torch.isnan(loss).any():
            continue
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss', loss.item(), j)


    torch.save(encoder.state_dict(), './weights/{}.pth'.format(logger_string))

if __name__ == "__main__":
    main()
