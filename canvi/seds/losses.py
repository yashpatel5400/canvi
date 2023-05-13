import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from utils import log_t_prior
from generate import generate_data_emulator

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



