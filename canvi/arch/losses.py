import torch
from generate import generate_data_favi
from utils import prior_t_sample_tup, transform_parameters, transform_parameters_batch
import torch.distributions as D
import torch.nn as nn
import numpy as np

def get_imp_weights(pts, num_samples, mdn=True, flow=False, log=False, prop_prior=0., **kwargs):
    mb_size = kwargs['mb_size']
    encoder = kwargs['encoder']
    K = num_samples
    device = kwargs['device']
    log_target = kwargs['log_target']
    log_prior_batch = kwargs['log_prior_batch']

    particles = encoder.sample(K, pts.float().to(device))
    rparticles = particles.reshape(K*mb_size, -1)
    repeated_pts = pts.repeat(K, 1, 1).transpose(0,1)
    repeated_pts = repeated_pts.reshape(K*mb_size, -1).to(device)
    log_denoms = encoder.log_prob(rparticles, repeated_pts)
    log_denoms = log_denoms.reshape(K, mb_size)
    log_nums = log_prior_batch(particles, **kwargs).reshape(K, mb_size).to(device) + log_target(particles, pts, K, **kwargs).reshape(K, mb_size).to(device)
    #log_nums = log_target(particles, repeated_pts, **kwargs).reshape(K, mb_size)
    log_weights = log_nums - log_denoms
    weights = nn.Softmax(0)(log_weights)
    if log:
        return particles, weights, log_weights
    else:
        return particles, weights

def iwbo_loss(x, **kwargs):
    mb_size = kwargs['mb_size']
    encoder = kwargs['encoder']
    K = kwargs['K']
    device = kwargs['device']

    # Choose data points
    indices = torch.randint(low=0, high=len(x), size=(mb_size,))
    pts = x[indices]

    particles, weights, log_weights = get_imp_weights(pts, num_samples=K, mdn=False, flow=True, log=True, **kwargs)
    return -1*torch.diag(weights.detach().T @ log_weights).mean()


def elbo_loss(x, **kwargs):
    mb_size = kwargs['mb_size']
    encoder = kwargs['encoder']
    K = 1 #override for elbo
    device = kwargs['device']

    # Choose data points
    indices = torch.randint(low=0, high=len(x), size=(mb_size,))
    pts = x[indices]

    particles, weights, log_weights = get_imp_weights(pts, num_samples=K, mdn=False, flow=True, log=True, **kwargs)
    return -1*torch.diag(weights.detach().T @ log_weights).mean()
    
def favi_loss(**kwargs):
    device = kwargs['device']
    encoder = kwargs['encoder']
    mb_size = kwargs['mb_size']

    z, x = generate_data_favi(mb_size, **kwargs)
    lps = encoder.log_prob(z.float().to(device), x.to(device).float())
    return -1*lps.mean()

def get_thetas_grid(mins, maxs, K = 200):
    # K -> discretization of the grid (assumed same for each dimension)
    d = len(mins) # dimensionality of theta
    ranges = [np.arange(mins[i], maxs[i], (maxs[i] - mins[i]) / K) for i in range(d)]
    dA = np.prod([(maxs[i] - mins[i]) / K for i in range(d)])
    theta_grid = np.array(np.meshgrid(*ranges)).T.astype(np.float32)
    return theta_grid.reshape(-1, theta_grid.shape[-1]), dA

def get_exact_vol(encoder, test_x, theta_mins, theta_maxs, conformal_quantile, device, my_t_priors):
    theta_grid, dA = get_thetas_grid(theta_mins, theta_maxs)
    test_x_tiled = torch.permute(torch.tile(test_x, (theta_grid.shape[0],1,1)), (1, 0, 2)).to(device)
    theta_grid_tiled = torch.tile(torch.Tensor(theta_grid), (test_x_tiled.shape[0],1,1)).to(device)

    flat_x = test_x_tiled.reshape(-1, test_x_tiled.shape[-1])
    flat_theta_grid = theta_grid_tiled.reshape(-1, theta_grid_tiled.shape[-1])
    flat_theta_grid[:,0] = my_t_priors[0].inv_transform(flat_theta_grid[:,0])
    flat_theta_grid[:,1] = my_t_priors[1].inv_transform(flat_theta_grid[:,1])
    
    probs = encoder.log_prob(flat_theta_grid, flat_x).detach().cpu().exp().numpy()
    exact_vol = (np.sum((1 / probs) < conformal_quantile) * dA) / len(test_x)
    return exact_vol

def mc_lebesgue(prior, encoder, test_x, conformal_quantile, device, kwargs, task_factor = 1/2, K = 10, S = 1_000):
    mc_set_size_est_ks = []
    for lambda_k in np.linspace(0, 0.1, K):
        # start = time.time() 
        zs = (torch.rand((len(test_x), S)) < lambda_k).to(device) # indicators for mixture draw

        prior_theta_dist = prior_t_sample_tup((len(test_x), S), **kwargs)
        empirical_theta_dist = encoder.sample((S), test_x)
        sample_x = torch.permute(torch.tile(test_x, (S,1,1)), (1, 0, 2))

        mixed_theta_dist = torch.zeros(empirical_theta_dist.shape).to(device)
        mixed_theta_dist[torch.where(zs == 0)] = prior_theta_dist[torch.where(zs == 0)]
        mixed_theta_dist[torch.where(zs == 1)] = empirical_theta_dist[torch.where(zs == 1)]
        
        flat_mixed_theta_dist = mixed_theta_dist.reshape(-1, mixed_theta_dist.shape[-1])
        flat_sample_x = sample_x.reshape(-1, sample_x.shape[-1])
        # flat_mixed_theta_dist = torch.Tensor(flat_mixed_theta_dist)

        # HACK: for problems with uniform priors, we can just manually compute prob
        # prior_probs = prior.log_prob(flat_mixed_theta_dist).detach().cpu().exp().numpy()
        var_probs = encoder.log_prob(flat_mixed_theta_dist, flat_sample_x).detach().cpu().exp().numpy()
        prior_probs = np.ones(var_probs.shape) * task_factor
        mixed_probs = (1 - lambda_k) * prior_probs + lambda_k * var_probs

        mc_set_size_est_k = np.mean((1 / var_probs < conformal_quantile).astype(float) / mixed_probs)
        mc_set_size_est_ks.append(mc_set_size_est_k)
    mc_est = np.mean(mc_set_size_est_ks)
    return mc_est

def lebesgue(cal_scores, theta_batch, x_batch, alpha, **kwargs):
    device = kwargs['device']
    encoder = kwargs['encoder']
    my_t_priors = kwargs['my_t_priors']
    
    conformal_quantile = np.quantile(cal_scores, q = 1-alpha)
    exact_area = get_exact_vol(encoder, x_batch, [-1, 0], [1, 1], conformal_quantile, device, my_t_priors)
    mc_area = mc_lebesgue(my_t_priors, encoder, x_batch, conformal_quantile, device, kwargs)

    return [exact_area], [mc_area]


