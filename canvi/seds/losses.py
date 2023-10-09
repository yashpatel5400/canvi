import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from utils import prior_t_sample, log_t_prior, prior_t_sample_tup
from generate import generate_data_emulator
import numpy as np


def get_dist(encoder, x, device):
    log_pi, mu, sigma = encoder(x.to(device))
    mix = D.Categorical(logits=log_pi)
    comp = D.Independent(D.Normal(mu, sigma), 1)
    return D.MixtureSameFamily(mix, comp)

def get_log_prob(encoder, x, theta, device):
    mixture = get_dist(encoder, x, device)
    return mixture.log_prob(theta).detach()

def get_imp_weights(pts, num_samples, mdn=True, flow=False, log=False, prop_prior=0., **kwargs):
    mb_size = kwargs['mb_size']
    encoder = kwargs['encoder']
    K = num_samples
    device = kwargs['device']
    log_target = kwargs['log_target']

    if mdn:
        log_pi, mu, sigma = encoder(pts.to(device))
        mix = D.Categorical(logits=log_pi)
        comp = D.Independent(D.Normal(mu, sigma), 1)
        mixture = D.MixtureSameFamily(mix, comp)

        # Our own rsample for elbo, iwbo
        particles = D.Normal(mu, sigma).rsample((K,))
        pparticles = particles.permute(-1, 0, 1, 2)
        to_keep = F.gumbel_softmax(log_pi.repeat(K,1,1), tau=1, hard=True)
        results = torch.mul(to_keep, pparticles).sum(-1)
        results = results.permute(1,2,0)

        log_nums = log_target(results, pts.to(device), **kwargs)
        log_denoms = mixture.log_prob(results)
        log_weights = log_nums - log_denoms
        weights = nn.Softmax(0)(log_weights).to(device)
        if log:
            return mixture, particles, weights, log_weights
        else:
            return mixture, particles, weights
    else:
        particles = encoder.sample(K, pts.float().to(device))
        rparticles = particles.reshape(K*mb_size, -1)
        repeated_pts = pts.repeat(K, 1, 1).transpose(0,1)
        repeated_pts = repeated_pts.reshape(K*mb_size, -1).to(device)
        log_denoms = encoder.log_prob(rparticles, repeated_pts)
        log_denoms = log_denoms.reshape(K, mb_size)
        log_nums = log_t_prior(particles, **kwargs).reshape(K, mb_size).to(device) + log_target(particles.transpose(0,1), pts, **kwargs).reshape(K, mb_size)
        #log_nums = log_target(particles, repeated_pts, **kwargs).reshape(K, mb_size)
        log_weights = log_nums - log_denoms
        weights = nn.Softmax(0)(log_weights)
        if log:
            return particles, weights, log_weights
        else:
            return particles, weights
        
def iwbo_loss(x, mdn=True, flow=False, **kwargs):
    assert not (mdn and flow), "One of mdn or flow flags must be false."
    mb_size = kwargs['mb_size']
    encoder = kwargs['encoder']
    K = kwargs['K']
    device = kwargs['device']

    # Choose data points
    indices = torch.randint(low=0, high=len(x), size=(mb_size,))
    pts = x[indices]

    if mdn:
        mixture, particles, weights, log_weights = get_imp_weights(pts, num_samples=K, mdn=True, flow=False, log=True, **kwargs)
        return -1*torch.diag(weights.detach().T @ log_weights).mean()
    elif flow:
        particles, weights, log_weights = get_imp_weights(pts, num_samples=K, mdn=False, flow=True, log=True, **kwargs)
        return -1*torch.diag(weights.detach().T @ log_weights).mean()


def elbo_loss(x, mdn=True, flow=False, **kwargs):
    assert not (mdn and flow), "One of mdn or flow flags must be false."
    mb_size = kwargs['mb_size']
    encoder = kwargs['encoder']
    K = 1 #override for elbo
    device = kwargs['device']

    # Choose data points
    indices = torch.randint(low=0, high=len(x), size=(mb_size,))
    pts = x[indices]

    if mdn:
        mixture, particles, weights, log_weights = get_imp_weights(pts, num_samples=K, mdn=True, flow=False, log=True, **kwargs)
        return -1*torch.diag(weights.detach().T @ log_weights).mean()
    elif flow:
        particles, weights, log_weights = get_imp_weights(pts, num_samples=K, mdn=False, flow=True, log=True, **kwargs)
        return -1*torch.diag(weights.detach().T @ log_weights).mean()
    

def favi_loss(mdn=True, flow=False, **kwargs):
    assert not (mdn and flow), "One of mdn or flow flags must be false."
    device = kwargs['device']
    K = kwargs['K']
    encoder = kwargs['encoder']
    n_pts = kwargs['n_pts']

    theta, x = generate_data_emulator(n_pts, return_theta=True, **kwargs)
    if flow:
        lps = encoder.log_prob(theta.float().to(device), x.to(device).float())
        return -1*lps.mean()
    elif mdn:
        neg_lps = encoder.loss(x.to(device).float(), theta.to(device).float())
        return neg_lps.mean()
    else:
        raise ValueError('At least one of mdn or flow flags must be true.')


def mc_lebesgue(prior, encoder, test_x, conformal_quantile, device, kwargs, task_factor = 1/2, K = 10, S = 1_000):
    mc_set_size_est_ks = []
    for lambda_k in np.linspace(0, 1, K):
        zs = (torch.rand(S) < lambda_k).to(device) # indicators for mixture draw

        prior_theta_dist = prior_t_sample(S, **kwargs)
        theta_dist = get_dist(encoder, test_x, device)
        empirical_theta_dist = theta_dist.sample((S,)).squeeze()

        mixed_theta_dist = torch.zeros(empirical_theta_dist.shape).to(device)
        mixed_theta_dist[torch.where(zs == 0)] = prior_theta_dist[torch.where(zs == 0)]
        mixed_theta_dist[torch.where(zs == 1)] = empirical_theta_dist[torch.where(zs == 1)]

        var_probs = theta_dist.log_prob(mixed_theta_dist).detach().cpu().exp().numpy()
        prior_probs = log_t_prior(mixed_theta_dist, **kwargs).detach().cpu().exp().numpy()
        mixed_probs = (1 - lambda_k) * prior_probs + lambda_k * var_probs

        mc_set_size_est_k = np.mean((1 / var_probs < conformal_quantile).astype(float) / mixed_probs)
        mc_set_size_est_ks.append(mc_set_size_est_k)
    return np.mean(mc_set_size_est_ks)

def lebesgue(cal_scores, theta_batch, x_batch, alpha, **kwargs):
    device = kwargs['device']
    encoder = kwargs['encoder']
    my_t_priors = kwargs['my_t_priors']
    
    conformal_quantile = np.quantile(cal_scores, q = 1-alpha)
    mc_area = mc_lebesgue(my_t_priors, encoder, x_batch, conformal_quantile, device, kwargs)
    return mc_area