import os
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
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
import torch.distributions as D
import torch
import numpy as np
import matplotlib.pyplot as plt
from cde.mdn import MixtureDensityNetwork
from cde.nsf import build_nsf, EmbeddingNet
from utils import resample, prior_t_sample, log_t_prior, transform_thetas
from losses import favi_loss, elbo_loss, iwbo_loss
from generate import generate_data_emulator, generate_data
from modules import TransformedFlatDirichlet, TransformedUniform, OurDesiMCMC, FAVIDataset, PROVABGSEmulator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def log_target(thetas, seds, **kwargs):
    '''
    Use chi_square based llk calculation from our generative model.
    Maybe figure out how this is implicitly defined.

    tthetas: (n_batch, 12) batch of thetas
    sed: a single SED

    Returns: (n_batch,) array of log likelihoods
    '''
    noise = kwargs['noise']
    multiplicative_noise = kwargs['multiplicative_noise']
    generator = kwargs['generator']
    scale = kwargs['scale']
    emulator = kwargs['emulator']

    means = emulator(thetas).clamp(min=0.)
    diffs = means - seds
    real_noise = torch.abs(means)*multiplicative_noise+1e-8
    multiplier = -.5*real_noise**(-2)
    results = torch.multiply(multiplier, torch.square(diffs)).sum(-1)

    return results.detach()


def setup(cfg):

    my_t_priors = [
        TransformedFlatDirichlet(dim=4),
        TransformedUniform(0., 1.),
        TransformedUniform(1e-2, 13.27),
        TransformedUniform(4.5e-5, 1.5e-2),
        TransformedUniform(4.5e-5, 1.5e-2),
        TransformedUniform(0., 3.),
        TransformedUniform(0., 3.),
        TransformedUniform(-2., 1.),
    ]

    sizes = [3,1,1,1,1,1,1,1]
    sizes_transformed = [4,1,1,1,1,1,1,1]
    jitter = cfg.data.jitter
    z_min = torch.tensor([-np.inf,-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    z_max = torch.tensor([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    K = cfg.smc.K
    n_pts = cfg.data.n_pts
    noise = cfg.data.noise
    multiplicative_noise = cfg.data.multiplicative_noise
    scale = cfg.data.scale
    smooth = cfg.data.smooth
    smooth_parameter = cfg.data.smooth_parameter
    obs_grid = np.arange(3000., 10000., 5.)
    generator = Models.NMF(burst=True, emulator=True)
    kwargs = {
        'K':K,
        'n_pts': n_pts,
        'sizes': sizes,
        'sizes_transformed': sizes_transformed,
        'jitter': jitter,
        'z_min': z_min,
        'z_max': z_max,
        'obs_grid': obs_grid,
        'my_t_priors': my_t_priors,
        'generator': generator,
        'noise': noise,
        'multiplicative_noise': multiplicative_noise,
        'smooth': smooth,
        'scale': scale,
        'smooth_parameter': smooth_parameter,
        'log_target': log_target,
    }

    # Generate data
    fake_thetas, fake_seds = generate_data(n_pts, True, **kwargs)
    
    epochs = cfg.training.epochs
    device=cfg.training.device
    mb_size = cfg.training.mb_size

    kwargs['mb_size'] = mb_size
    kwargs['device'] = device

    z_dim = fake_thetas.shape[-1]
    x_dim = fake_seds.shape[-1]

    # Set up encoder
    if cfg.encoder.type == 'flow':
        # EXAMPLE BATCH FOR SHAPES
        num_obs_flow = K*mb_size
        fake_zs = torch.randn((K*mb_size, z_dim))
        fake_xs = torch.randn((K*mb_size, x_dim))
        encoder = build_nsf(fake_zs, fake_xs, z_score_x='none', z_score_y='none', hidden_features=32, embedding_net=EmbeddingNet(len(obs_grid)).float())
        # encoder2.log_prob(particles.squeeze(1).float().to(device), pts.float().to(device).repeat(K,1))
    elif cfg.encoder.type == 'mdn':
        encoder = MixtureDensityNetwork(dim_in=len(obs_grid), dim_out=sum(sizes), n_components=20, hidden_dim=512)
    else:
        raise ValueError('cfg.encoder.type must be one of "flow", "mdn"')
    
    # for p in encoder.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -1e-2, 1e-2))
    
    # Set up emulator 
    emulator = PROVABGSEmulator(dim_in=z_dim, dim_out=x_dim)
    emulator.load_state_dict(torch.load('./emulator_weights/weights_min=10,max=11,epochs=2000'))
    emulator.to(device)
    kwargs['emulator'] = emulator

    thetas, seds = generate_data_emulator(n_pts, True, **kwargs)
    
    # Set up logging string
    kwargs['encoder'] = encoder
    name = 'flow' if cfg.encoder.type == 'flow' else 'mdn'
    if name == 'flow':
        mdn = False
        flow = True
    else:
        mdn = True
        flow = False
    logger_string = '{},{},{},{},noise={},mult={},smooth={}'.format(cfg.training.loss, name, cfg.training.lr, K, noise,multiplicative_noise, smooth)
    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.training.lr)
    
    # Select loss function
    loss_name = cfg.training.loss
    kwargs['loss'] = loss_name
    # if loss_name == 'elbo':
    #     loss_fcn = lambda: elbo_loss(seds, mdn=mdn, flow=flow, **kwargs)
    # elif loss_name == 'iwbo':
    #     loss_fcn = lambda: iwbo_loss(seds, mdn=mdn, flow=flow, **kwargs)
    # elif loss_name == 'favi':
    #     loss_fcn = lambda: favi_loss(mdn=mdn, flow=flow, **kwargs)
    # else:
    #     raise ValueError('Specify an appropriate loss name string.')
    
    return (
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
    )
