import torch
import torch.distributions as D
import math
import os
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

def generate_data(n_obs, innov=None, **kwargs):
    prior = kwargs['prior']
    T = kwargs['T']
    device = kwargs['device']

    thetas = prior.sample((n_obs,)).to(device)
    theta1 = thetas[:,0]
    theta2 = thetas[:,1]

    ys = torch.empty((n_obs, T)).to(device)
    es = torch.empty((n_obs, T)).to(device)
    noise = D.Normal(0., 1.)
    innovations = noise.sample((n_obs,T)).to(device) if not innov else innov.to(device)
    ys[:,0] = 0.
    es[:,0] = 0.#innovations[:,0]
    for i in range(1,T):
        ei = innovations[:,i]*torch.sqrt(.2+theta2*(es[:,i-1]**2))
        yi = theta1*ys[:,i-1]+ei
        ys[:,i] = yi
        es[:,i] = ei
    return thetas, ys