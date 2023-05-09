import torch
from generate import generate_data
from utils import transform_parameters
import torch.distributions as D
import torch.nn as nn

def get_imp_weights(pts, num_samples, mdn=True, flow=False, log=False, prop_prior=0., **kwargs):
    mb_size = kwargs['mb_size']
    encoder = kwargs['encoder']
    K = num_samples
    device = kwargs['device']
    log_target = kwargs['log_target']
    log_prior = kwargs['log_prior']

    particles, log_denoms = encoder.sample_and_log_prob(num_samples=K, context=pts.float().to(device))
    particles = particles.reshape(K*mb_size, -1)
    log_denoms = log_denoms.view(K,-1)
    #repeated_pts = pts.repeat(K, 1, 1).reshape(K*mb_size, -1).to(device)
    log_nums = log_prior(particles, **kwargs).reshape(K, mb_size).to(device) + log_target(particles, pts.to('cpu'), num_samples, **kwargs).reshape(K, mb_size).to(device)
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

    z, x = generate_data(mb_size, **kwargs)
    lps = encoder.log_prob(z.float().to(device), x.to(device).float())
    return -1*lps.mean()

def favi_loss_mdn( **kwargs):
    device = kwargs['device']
    encoder = kwargs['encoder']
    mb_size = kwargs['mb_size']

    z, x = generate_data(mb_size, **kwargs)
    log_pi, mu, sigma = encoder(x.to(device))
    mix = D.Categorical(logits=log_pi)
    comp = D.Independent(D.Normal(mu, sigma), 1)
    mixture = D.MixtureSameFamily(mix, comp)

    lps = mixture.log_prob(z.to(device))
    return -1*lps.mean()



