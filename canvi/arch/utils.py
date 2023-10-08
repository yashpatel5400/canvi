import torch
import math
import torch.distributions as D
import torch.nn as nn
import numpy as np
from scipy.integrate import dblquad
import scipy.integrate as integrate
from scipy.integrate import quad

# def integrand(eps, theta1, theta2, **kwargs):
#     lead = 1/math.sqrt(2*math.pi*(.2+theta2*(eps**2)))
#     e1 = np.exp(-1*(eps**2)/(2*(.2+theta2*(eps**2))))
#     follow = 1/math.sqrt(2*math.pi)
#     e2 = np.exp(-1*(eps**2)/2)
#     return lead*e1*follow*e2

# def get_1(theta1, theta2, **kwargs):
#     I = quad(integrand, -10, 10, args=(theta1,theta2))[0]
#     return I

def density(eps_vec, theta1, theta2, x, **kwargs):
    #lead = get_1(theta1, theta2, **kwargs)
    T = len(eps_vec)
    running_sum = 0.
    for i in range(1, T):
        const = 1/math.sqrt(2*math.pi*(.2+theta2*(eps_vec[i-1]**2)))
        const = math.log(const)
        fact = -1*(eps_vec[i]**2)/(2*(.2+theta2*eps_vec[i-1]**2))
        running_sum += const
        running_sum += fact
    term = math.exp(running_sum)
    return term

def integrand2(theta1, theta2, x, **kwargs):
    T = kwargs['T']
    Q = np.eye(T) + np.diag(-1*theta1*np.ones(T-1), -1)
    eps_vec = Q @ x.reshape(-1,1)
    eps_vec = eps_vec.reshape(-1)
    return density(eps_vec, theta1, theta2, x, **kwargs)

def normalizing_integral(x, **kwargs):
    func_to_integrate = lambda theta1, theta2: integrand2(theta1, theta2, x, **kwargs)
    return dblquad(func_to_integrate, 0., 1., lambda theta2: -1, lambda theta2: 1)


def posterior(theta1, theta2, x, **kwargs):
    T = kwargs['T']
    Q = np.eye(T) + np.diag(-1*theta1*np.ones(T-1), -1)
    eps_vec = (Q @ x.reshape(-1,1)).reshape(-1)
    return density(eps_vec, theta1, theta2, x, **kwargs)

def prior_t_sample(n_obs, **kwargs):
    my_t_priors = kwargs['my_t_priors']
    samples = [prior.sample((n_obs,)) for prior in my_t_priors]
    samples = [x.unsqueeze(-1) if len(x.shape) == 1 else x for x in samples]
    #samples[1] = shrink_dirichlet(samples[1])
    return torch.cat(samples, -1)

def transform_parameters(theta, **kwargs):
    my_t_priors = kwargs['my_t_priors']
    new = torch.empty(theta.shape[0], 2)
    for j in range(len(my_t_priors)):
        new[:,j] = my_t_priors[j].transform(theta[:,j])
    return new

def transform_parameters_batch(theta, **kwargs):
    my_t_priors = kwargs['my_t_priors']
    new = torch.empty(theta.shape[0], theta.shape[1], 2)
    for j in range(len(my_t_priors)):
        new[...,j] = my_t_priors[j].transform(theta[...,j])
    return new

def log_target(particles, context, num_particles, **kwargs):
    T = kwargs['T']
    device = kwargs['device']
    K = num_particles
    params = transform_parameters_batch(particles, **kwargs).to(device)
    theta1 = params[...,0]
    theta2 = params[...,1]
    eyes = torch.eye(T).repeat(K,1,1).to(device) #batch of Qs
    repeated = -1*theta1.reshape(-1,1).repeat(1, T-1)
    subdiags = torch.diag_embed(repeated, -1)
    Qs = eyes + subdiags
    
    targets = context.reshape(-1,1).repeat(K, 1, 1)
    eps_vecs = torch.bmm(Qs, targets)
    eps_vecs = eps_vecs.squeeze(-1)

    running_sum = torch.zeros(1, K).to(device)
    for i in range(1, T):
        const = 1/torch.sqrt(2*math.pi*(.2+theta2*(eps_vecs[:,i-1]**2)))
        const = torch.log(const).to(device)
        fact = -1*(eps_vecs[:,i]**2)/(2*(.2+theta2*eps_vecs[:,i-1]**2))
        running_sum += const
        running_sum += fact
    return running_sum

def log_prior(particles, **kwargs):
    prior = kwargs['prior']
    theta = transform_parameters(particles, **kwargs)
    lps = prior.log_prob(theta) #
    return lps

def log_prior_batch(particles, **kwargs):
    prior = kwargs['prior']
    theta = transform_parameters_batch(particles, **kwargs)
    lps = prior.log_prob(theta) #
    return lps

