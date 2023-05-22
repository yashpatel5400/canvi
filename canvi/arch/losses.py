import torch
from generate import generate_data_favi
from utils import transform_parameters, transform_parameters_batch
import torch.distributions as D
import torch.nn as nn

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

def lebesgue(cal_scores, theta_batch, x_batch, alpha, **kwargs):
    device = kwargs['device']
    encoder = kwargs['encoder']
    my_t_priors = kwargs['my_t_priors']
    cal_scores = cal_scores.to(device)
    theta1vals = torch.arange(-1.+0.01, 1., .01)
    theta2vals = torch.arange(0.+0.01, 1., .01)
    X, Y = torch.meshgrid(theta1vals, theta2vals)
    eval_pts = torch.cartesian_prod(theta1vals, theta2vals).to(device)
    eval_pts_uncon = torch.empty(eval_pts.shape)
    eval_pts_uncon[:,0] = my_t_priors[0].inv_transform(eval_pts[:,0])
    eval_pts_uncon[:,1] = my_t_priors[1].inv_transform(eval_pts[:,1])

    # Get quantile
    q = torch.tensor([1-alpha]).to(device)
    quantiles = torch.quantile(cal_scores, q, dim=0)

    areas = []
    for j in range(x_batch.shape[0]):
        all_lps = encoder.log_prob(eval_pts_uncon, x_batch[j].reshape(1,-1).repeat(eval_pts_uncon.shape[0], 1))
        all_scores = -1*all_lps
        in_region = (all_scores < quantiles[0]).long().float().sum()
        area = in_region*(.01)*(.01)
        areas.append(area.item())


    return areas


