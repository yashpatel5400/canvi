import torch
import numpy as np
import torchvision
from torchvision import models
import random
from torchquad import MonteCarlo, Trapezoid, set_up_backend, enable_cuda
import torch.distributions as D
import math
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import statsmodels.api as sm
import scipy.integrate as integrate
from scipy.integrate import quad
from model_encoder import Encoder


torch.manual_seed(483972)
random.seed(483972)
np.random.seed(483972)

device = 'cpu'
trained_encoder = Encoder(device)
trained_encoder.load_state_dict(torch.load('trained_encoder.pth'))
trained_encoder.to(device)

# ASSESS COVERAGE RATE FROM FAVI
# For 10000 trials, compute 95% marginal CIs from the bivariate normal approximate posterior.
# Check for inclusion of both ground truth theta_1, theta_2 in these intervals.
num_trials = 10000
prior = D.Independent(D.Uniform(torch.tensor([-1., 0.]), torch.tensor([1., 1.])), 1)
def generate_data(theta1, theta2, T=100):
    noise = D.Normal(0., 1.)
    es = torch.empty((T,))
    es[0] = noise.sample()
    ys = torch.empty((T,))
    ys[0] = 0.
    innovations = noise.sample((T,))
    for i in range(1, T):
        ei = innovations[i]*math.sqrt(.2+theta2*(es[i-1]**2))
        yi = theta1*ys[i-1]+ei
        ys[i] = yi.item()
        es[i] = ei.item()
    return ys, es

successes1 = 0 #parameter 1
successes2 = 0 #parameter 2
lens1favi = [] #confidence interval lengths
lens2favi = [] #confidence intervals lengths
for j in range(num_trials):
    if j % 1000 == 0:
        print('On trial {}'.format(j))
    theta = prior.sample().to(device)
    theta1, theta2 = theta[0].item(), theta[1].item()
    draw_fake = generate_data(theta1, theta2)[0].to(device)
    mean_, cov_ = trained_encoder(draw_fake)

    # First parameter
    z_mult = 1.96
    left, right = mean_[0] - z_mult*torch.sqrt(cov_[0,0]), mean_[0] + z_mult*torch.sqrt(cov_[0,0])
    check1 = (theta1 <= right) & (theta1 >= left)
    if check1:
        successes1 += 1
    lens1favi.append(min(right, torch.tensor(1.))-max(left, torch.tensor(-1.)))

    # Second parameter
    z_mult = 1.96
    left, right = mean_[1] - z_mult*torch.sqrt(cov_[1,1]), mean_[1] + z_mult*torch.sqrt(cov_[1,1])
    check2 = (theta2 <= right) & (theta2 >= left)
    if check2:
        successes2 += 1
    lens2favi.append(min(right, torch.tensor(1.))-max(left, torch.tensor(0.)))

print('FAVI SUCCESS RATES:\n {} \n {}'.format(successes1/num_trials, successes2/num_trials))

# ASSESS COVERAGE RATE FROM CANVI

# First, get the scores
validation_size = 10000
scores1 = []
scores2 = []
for i in range(validation_size):
    theta = prior.sample().to(device)
    theta1, theta2 = theta[0].item(), theta[1].item()
    draw_fake = generate_data(theta1, theta2)[0].to(device)
    mean_, cov_ = trained_encoder(draw_fake)
    
    # Scores
    s1 = torch.abs((theta1 - mean_[0]))/torch.sqrt(cov_[0,0])
    s2 = torch.abs((theta2 - mean_[1]))/torch.sqrt(cov_[1,1])

    scores1.append(s1.item())
    scores2.append(s2.item())

scores1 = np.array(scores1)
scores2 = np.array(scores2)
quantiles1 = np.quantile(scores1, [.95])[0]
quantiles2 = np.quantile(scores2, [.95])[0]

successes1 = 0
successes2 = 0
lens1canvi = []
lens2canvi = []
for j in range(num_trials):
    if j % 1000 == 0:
        print('On trial {}'.format(j))
    theta = prior.sample().to(device)
    theta1, theta2 = theta[0].item(), theta[1].item()
    draw_fake = generate_data(theta1, theta2)[0].to(device)
    mean_, cov_ = trained_encoder(draw_fake)

    # First parameter
    z_mult = quantiles1
    left, right = mean_[0] - z_mult*torch.sqrt(cov_[0,0]), mean_[0] + z_mult*torch.sqrt(cov_[0,0])
    check1 = (theta1 <= right) & (theta1 >= left)
    if check1:
        successes1 += 1
    lens1canvi.append(min(right, torch.tensor(1.))-max(left, torch.tensor(-1.)))

    # Second parameter
    z_mult = quantiles2
    left, right = mean_[1] - z_mult*torch.sqrt(cov_[1,1]), mean_[1] + z_mult*torch.sqrt(cov_[1,1])
    check2 = (theta2 <= right) & (theta2 >= left)
    if check2:
        successes2 += 1
    lens2canvi.append(min(right, torch.tensor(1.))-max(left, torch.tensor(0.)))
    

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

print('CANVI SUCCESS RATES:\n {} \n {}'.format(successes1/num_trials, successes2/num_trials))

import matplotlib.pyplot as plt
lens1favi = torch.stack(lens1favi).detach().cpu().numpy()
lens2favi = torch.stack(lens2favi).detach().cpu().numpy()
lens1canvi = torch.stack(lens1canvi).detach().cpu().numpy()
lens2canvi = torch.stack(lens2canvi).detach().cpu().numpy()
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

ax1.set_title('CI Lengths: $\\theta_1$')
ax1.set_ylabel('Lengths')
ax1.violinplot([lens1favi, lens1canvi])

ax2.set_title('CI Lengths: $\\theta_2$')
ax2.set_ylabel('Lengths')
ax2.violinplot([lens2favi, lens2canvi])

# set style for the axes
labels = ['FAVI', 'CANVI']
for ax in [ax1, ax2]:
    set_axis_style(ax, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.savefig('arch1lengths.png')