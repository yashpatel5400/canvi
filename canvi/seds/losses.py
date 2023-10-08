import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from utils import log_t_prior
from generate import generate_data_emulator
import numpy as np

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


def mc_lebesgue(priors, encoder, test_x, conformal_quantile, K = 10, S = 1_000):
    mc_set_size_est_ks = []
    for k in np.linspace(0, 1, K):
        # start = time.time() 
        lambda_k = 1 - k / K
        zs = np.random.random((len(test_x), S)) < lambda_k # indicators for mixture draw

        prior_theta_dist = np.concatenate([prior.sample((len(test_x), S)).unsqueeze(-1).detach().cpu().numpy() for prior in priors], axis=-1)
        empirical_theta_dist = encoder.sample(S, test_x).detach().cpu().numpy()
        sample_x = np.transpose(np.tile(test_x, (S,1,1)), (1, 0, 2))

        mixed_theta_dist = np.zeros(empirical_theta_dist.shape)
        mixed_theta_dist[np.where(zs) == 0] = prior_theta_dist[np.where(zs) == 0]
        mixed_theta_dist[np.where(zs) == 1] = empirical_theta_dist[np.where(zs) == 1]

        flat_mixed_theta_dist = mixed_theta_dist.reshape(-1, mixed_theta_dist.shape[-1])
        flat_sample_x = sample_x.reshape(-1, sample_x.shape[-1])
        flat_mixed_theta_dist = torch.Tensor(flat_mixed_theta_dist)
        
        # prior_probs = prior.log_prob(flat_mixed_theta_dist).detach().cpu().exp().numpy()
        var_probs = (-encoder.loss(flat_mixed_theta_dist, flat_sample_x)).detach().cpu().exp().numpy()
        prior_probs = np.ones(var_probs.shape) * 1 / 2
        mixed_probs = lambda_k * prior_probs + (1 - lambda_k) * var_probs

        mc_set_size_est_k = np.mean((1 / var_probs < conformal_quantile).astype(float) / mixed_probs)
        mc_set_size_est_ks.append(mc_set_size_est_k)
    mc_est = np.mean(mc_set_size_est_ks)
    return mc_est

def lebesgue(cal_scores, theta_batch, x_batch, alpha, **kwargs):
    device = kwargs['device']
    encoder = kwargs['encoder']
    my_t_priors = kwargs['my_t_priors']
    cal_scores = cal_scores.to(device)
    
    # Get quantile
    q = torch.tensor([1-alpha]).to(device)
    quantiles = torch.quantile(cal_scores, q, dim=0)

    mc_areas = []
    for j in range(x_batch.shape[0]):
        test_x = x_batch[j].reshape(1,-1)
        mc_area = mc_lebesgue(my_t_priors, encoder, test_x, quantiles[0].cpu().detach().numpy())
        mc_areas.append(mc_area)

    return mc_areas