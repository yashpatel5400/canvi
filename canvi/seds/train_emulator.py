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
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import random
import torch.nn as nn
from generate import generate_data_deterministic
from modules import PROVABGSEmulator
from setup import setup


@hydra.main(version_base=None, config_path="../conf", config_name="config3")
def main(cfg : DictConfig) -> None:
    # initialize(config_path="../conf", job_name="test_app")
    # cfg = compose(config_name="config3")
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)
    cfg.smc.only=True


    (
        thetas,
        seds,
        kwargs
    ) = setup(cfg)

    dim_in = thetas.shape[-1]
    dim_out = seds.shape[-1]
    epochs = 2000

    device = 'cuda:7'
    emulator = PROVABGSEmulator(dim_in=dim_in, dim_out=dim_out).to(device)
    optimizer = torch.optim.Adam(emulator.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for j in range(epochs):
        if j % 500 == 0:
            print('On epoch {}.'.format(j))
        syn_theta, syn_x = generate_data_deterministic(1000, return_theta=True, **kwargs)
        syn_theta = syn_theta.float().to(device)
        syn_x = syn_x.float().to(device)
        x_hat = emulator.forward(syn_theta).float()
        optimizer.zero_grad()
        loss = criterion(x_hat, syn_x)
        loss.backward()
        optimizer.step()

        print('Loss {} is {}.'.format(j, loss.item()))
        del loss

        if j % 100 == 0:
            torch.save(emulator.state_dict(), './exp3/emulator_weights/weights_min={},max={},epochs={}'.format(cfg.data.min_mag, cfg.data.max_mag, j))


    emulator.load_state_dict(torch.load('./exp3/emulator_weights/weights_min=10,max=11,epochs=100000'))
    

if __name__ == "__main__":
    main()