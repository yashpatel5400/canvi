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
import corner as DFM
import numpy as np
from utils import transform_thetas, put_to_right_form
from modules import OurDesiMCMC
from utils import transform_thetas
from setup import setup
from modules import TransformedFlatDirichlet, TransformedUniform, OurDesiMCMC, FAVIDataset
###------------MCMC Code--------####
def plot_corner_flow(flow, true_theta, sed, n_samples=10000, **kwargs):
    '''
    Given an SMC object, samples from it and produces a corner plot.
    '''
    samples, lps = flow.sample_and_log_prob(n_samples, sed.view(1,-1).float())
    thetas = samples.squeeze(0)
    tthetas = transform_thetas(thetas, **kwargs)
    truth = transform_thetas(true_theta.view(1,-1), **kwargs).view(-1)
    fig = DFM.corner(tthetas.detach().cpu().numpy(), color='red',
                     labels=[r'$\log M_*$', r'$\beta_1$',  r'$\beta_2$',  r'$\beta_3$',  r'$\beta_4$', 
                       r'$f_{\rm burst}$', r'$t_{\rm burst}$', r'$\gamma_1$', r'$\gamma_2$', r'$\tau_{\rm BC}$', r'$\tau_{\rm ISM}$', r'$n_{\rm dust}$'],
                       label_kwargs={'fontsize': 14},
                       truths=truth,
                       hist_kwargs={'density': True})
    return fig

def run_mcmc_and_plot(flow, true_theta, sed, n_samples=10000, **kwargs):
    obs_grid = kwargs['obs_grid']
    noise = kwargs['noise']
    ivar = noise**(-2)

    nwalkers=30    # number of MCMC walkers
    start = put_to_right_form(true_theta.view(1,-1), False, **kwargs)
    theta_start = start.repeat((nwalkers, 1)).cpu().numpy()
    prior = Infer.load_priors([
            Infer.UniformPrior(10., 11., label='sed'),
            Infer.FlatDirichletPrior(4, label='sed'), 
            Infer.UniformPrior(0., 1., label='sed'), 
            Infer.UniformPrior(1e-2, 13.27, label='sed'), 
            Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), 
            Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), 
            Infer.UniformPrior(0., 3., label='sed'), 
            Infer.UniformPrior(0., 3., label='sed'),  
            Infer.UniformPrior(-2., 1., label='sed') 
        ])
    theta_start2 = np.stack([prior.sample() for k in range(nwalkers)])
    jitter = 1e-6
    jit = D.Normal(0., jitter).sample(theta_start.shape).cpu().numpy()

    # declare SPS model
    m_nmf = Models.NMF(burst=True, emulator=True)
    mcmc = OurDesiMCMC(model=m_nmf, prior=prior, kwargs=kwargs)
    chain = mcmc.run(
            wave_obs=obs_grid, # observed wavelength
            flux_obs=sed, # observed flux of spectrum
            flux_ivar_obs=ivar*np.ones(len(sed)), # no noise in this example
            zred=0.,       # redshift
            vdisp=0.,       # velocity dispersion (set to 0 for simplicity)
            sampler='zeus', # zeus ensemble slice sample
            nwalkers=nwalkers,    # number of MCMC walkers
            burnin=500,     # burn in iterations 
            opt_maxiter=2000, # maximum number of iterations for initial optimizer
            niter=1000,     # number of iterations after burn in
            progress=True,
            theta_start=(theta_start+jit))  # show progress bar
    
    best_fit_theta = chain['theta_bestfit']  
    truth = transform_thetas(true_theta.view(1,-1), **kwargs).view(-1)
    samples = chain['mcmc_chain'] #n_iter x n_walkers x dim
    flattened = UT.flatten_chain(samples)

    fig = plt.figure(figsize=(15, 18))
    _ = DFM.corner(flattened, fig=fig,
                labels=[r'$\log M_*$', r'$\beta_1$',  r'$\beta_2$',  r'$\beta_3$',  r'$\beta_4$', 
                       r'$f_{\rm burst}$', r'$t_{\rm burst}$', r'$\gamma_1$', r'$\gamma_2$', r'$\tau_{\rm BC}$', r'$\tau_{\rm ISM}$', r'$n_{\rm dust}$'],
                label_kwargs={'fontsize': 14},
                truths = truth,
                hist_kwargs={'density': True})
    samples, lps = flow.sample_and_log_prob(n_samples, sed.view(1,-1).float())
    thetas = samples.squeeze(0)
    tthetas = transform_thetas(thetas, **kwargs)
    _ = DFM.corner(tthetas.detach().cpu().numpy(), color='red', fig=fig,
                     labels=[r'$\log M_*$', r'$\beta_1$',  r'$\beta_2$',  r'$\beta_3$',  r'$\beta_4$', 
                       r'$f_{\rm burst}$', r'$t_{\rm burst}$', r'$\gamma_1$', r'$\gamma_2$', r'$\tau_{\rm BC}$', r'$\tau_{\rm ISM}$', r'$n_{\rm dust}$'],
                       label_kwargs={'fontsize': 14},
                       hist_kwargs={'density': True})
    
    return fig

@hydra.main(version_base=None, config_path="../conf", config_name="config3")
def main(cfg : DictConfig) -> None:
    initialize(config_path="../conf", job_name="test_app")
    cfg = compose(config_name="config3")
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    cfg.smc.only = False

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
        writer,
        optimizer,
        loss_fcn,
        kwargs
    ) = setup(cfg)

    # Load state dict
    encoder.load_state_dict(torch.load('exp3/weights/weights_{}'.format(logger_string)))
    encoder = encoder.to('cpu')
    encoder.eval()

    for j in cfg.plots.flow.points:
        test_sed = seds[j].cpu()
        test_truths = thetas[j].cpu()
        fig = plot_corner_flow(encoder, test_truths, test_sed, n_samples=cfg.plots.n_samples, **kwargs)
        plt.savefig('./exp3/plots/flow_{}_{}.png'.format(j, logger_string))
        plt.clf()

    for j in cfg.plots.flow_and_mcmc.points:
        test_sed = seds[j].cpu()
        test_truths = thetas[j].cpu()
        fig = run_mcmc_and_plot(encoder, test_truths, test_sed, n_samples=10000, **kwargs)
        plt.savefig('./exp3/plots/mcmcflow_{}_{}.png'.format(j, logger_string))

if __name__ == "__main__":
    main()


