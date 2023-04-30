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
from generate import generate_data
import torch
from utils import prior_t_sample, resample, transform_thetas
import torch.distributions as D

def assess_calibration(thetas, x, logger_string, mdn=True, flow=False, n_samples=1000, alpha=.05, **kwargs):
    assert not (mdn and flow), "One of mdn or flow flags must be false."
    encoder = kwargs['encoder']
    device = kwargs['device']

    results = torch.zeros_like(thetas[0])
    for j in range(x.shape[0]):
        true_param = thetas[j]
        observation = x[j]
        # Sample from encoder
        if mdn:
            log_pi, mu, sigma = encoder(x[j].to(device))
            mix = D.Categorical(logits=log_pi.view(-1))
            comp = D.Independent(D.Normal(mu.squeeze(0), sigma.squeeze(0)), 1)
            mixture = D.MixtureSameFamily(mix, comp)
            particles = mixture.sample((n_samples,)).clamp(-1., 1.)
        elif flow:
            particles, log_denoms = encoder.sample_and_log_prob(num_samples=n_samples, context=x[j].view(1,-1).float().to(device))
        particles = particles.reshape(n_samples, -1)
        q = torch.tensor([alpha/2, 1-alpha/2])
        quantiles = torch.quantile(particles, q, dim=0)
        success = ((true_param > quantiles[0]) & (true_param < quantiles[1])).long()
        results += success

    return results/x.shape[0]

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
    
    test_thetas, test_seds = generate_data(n_samples=1000, return_theta=True, **kwargs)
    results = assess_calibration(test_thetas, test_seds, logger_string, mdn=False, flow=True, n_samples=10000, alpha=0.5, **kwargs)
    print(results)



    results = assess_calibration(thetas, seds, logger_string, mdn=False, flow=True, n_samples=10000, alpha=0.25, **kwargs)
    print(results)


if __name__ == "__main__":
    main()