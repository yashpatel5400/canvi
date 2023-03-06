import pickle
import os
import glob
import hypothesis as h
import numpy as np
import torch
import time

from sbi.inference.base import infer
from hypothesis.benchmark.tractable_small import Prior
from hypothesis.benchmark.tractable_small import Simulator
from hypothesis.stat import highest_density_level
from sbi.inference import MCABC
from tqdm import tqdm
from scipy.stats import gaussian_kde

prior = Prior()
extent = [ # I know, this isn't very nice :(
    prior.low[0].item(), prior.high[0].item(),
    prior.low[1].item(), prior.high[1].item()]


def build_posterior(simulation_budget, out, task_index, num_workers):
    prior = Prior()
    simulator = Simulator()
    min_samples_to_keep = 100.
    min_quantile = min_samples_to_keep/simulation_budget
    quantile = max([min_quantile, 0.01])

    theta_0 = prior.sample((1,))
    x_0 = simulator(theta_0)

    inference = MCABC(simulator, prior, num_workers=num_workers)
    posterior = inference(x_0, simulation_budget, quantile=quantile)

    # Better way?
    theta_accepted = posterior._samples
    posterior = gaussian_kde(np.swapaxes(theta_accepted.numpy(), 0, 1))

    with open(os.path.join(out, "posterior.pkl"), "wb") as handle:
        pickle.dump(posterior, handle)

    with open(os.path.join(out, "x_0.pkl"), "wb") as handle:
        pickle.dump(x_0, handle)

    with open(os.path.join(out, "theta_0.pkl"), "wb") as handle:
        pickle.dump(theta_0, handle)


@torch.no_grad()
def load_estimators_parameters_observables(query):
    paths = glob.glob(query)
    posteriors = [pickle.load(open(os.path.join(path, "posterior.pkl"), "rb")) for path in paths]
    parameters = [pickle.load(open(os.path.join(path, "theta_0.pkl"), "rb")).to(h.accelerator) for path in paths]
    observables = [pickle.load(open(os.path.join(path, "x_0.pkl"), "rb")).to(h.accelerator) for path in paths]

    return posteriors, parameters, observables

@torch.no_grad()
def compute_log_posterior(posterior, observable, resolution=100):
    # Prepare grid
    p1 = torch.linspace(extent[0], extent[1], resolution)
    p2 = torch.linspace(extent[2], extent[3], resolution)
    p1 = p1.to(h.accelerator)
    p2 = p2.to(h.accelerator)
    g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1))
    # Vectorize
    inputs = torch.cat([g1.reshape(-1, 1), g2.reshape(-1, 1)], dim=1)

    observable = observable.to(h.accelerator)

    inputs = np.swapaxes(inputs.numpy(), 0, 1)
    log_posterior = posterior.logpdf(inputs)
    #log_posterior = torch.stack([posterior.log_prob(inputs[i, :]) for i in range(len(inputs))], axis=0)
    assert (log_posterior.shape == (resolution**2,))

    return log_posterior

@torch.no_grad()
def coverage(posteriors, nominals, observables, alphas=[0.05]):
    n = len(nominals)
    covered = [0 for _ in alphas]

    for posterior, nominal, observable in tqdm(zip(posteriors, nominals, observables), "Coverages evaluated"):
        pdf = np.exp(compute_log_posterior(posterior, observable))
        #pdf = compute_log_posterior(posterior, observable).exp()
        nominal_pdf = np.exp(posterior.logpdf(np.swapaxes(nominal.numpy(), 0, 1)))
        #nominal_pdf = posterior.log_prob(nominal.squeeze()).exp()
        for i, alpha in enumerate(alphas):
            level = highest_density_level(pdf, alpha)
            if nominal_pdf >= level:
                covered[i] += 1

    return [x / n for x in covered]

@torch.no_grad()
def mutual_information(posteriors, nominals, observables):
    prior = Prior()
    n = len(nominals)
    mi = 0

    for posterior, nominal, observable in tqdm(zip(posteriors, nominals, observables), "Mutual information evaluated"):
        #log_posterior = posterior.log_prob(nominal.squeeze())
        log_posterior = torch.Tensor(posterior.logpdf(np.swapaxes(nominal.numpy(), 0, 1)))
        log_prior = prior.log_prob(nominal)
        log_r = log_posterior - log_prior
        mi += log_r

    return mi/n
