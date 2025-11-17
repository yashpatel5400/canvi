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

def generate_data(theta1, theta2, T=100):
    '''
    An ARCH(1) model, following Section 5.2 of
    https://arxiv.org/pdf/1611.10242.pdf
    '''
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

class Encoder(nn.Module):
    '''
        Given x, a time-series of population size (I think),
        computes the first 5 autocorrelation lags and their interactions as
        summary statistics. Results in 5 + (5 choose 2) = 15 features, input
        into a dense network which approximates an amoritzed posterior by a 
        bivariate normal with diagonal covariance matrix.
        '''
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.device = device
        self.enc = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
  
    def forward(self, x):
        lags = sm.tsa.acf(x.cpu(), nlags=5)[1:]
        a = lags[:-1]*lags[-1]
        b = lags[:-2]*lags[-2]
        c = lags[:-3]*lags[-3]
        d = lags[:-4]*lags[-4]
        lags, a, b, c, d = torch.tensor(lags), torch.tensor(a), torch.tensor(b), torch.tensor(c), torch.tensor(d)
        net_in = torch.cat([lags,a,b,c,d]).float().to(self.device)
        net_in = net_in.view(1,-1)
        out = self.enc(net_in).flatten()
        means = out[:2]
        cov = torch.diag(torch.exp(out[2:]))
        return means, cov


def main():
    torch.manual_seed(483972)
    random.seed(483972)
    np.random.seed(483972)


    prior = D.Independent(D.Uniform(torch.tensor([-1., 0.]), torch.tensor([1., 1.])), 1)

    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(6)
    set_up_backend("torch", data_type="float64")

    def loss_fcn(vec, mean, cov):
        distr = D.MultivariateNormal(mean, cov)
        return -1*(distr.log_prob(vec))
        
    encoder = Encoder(device).float().to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    epochs = 10000
    for i in range(epochs):
        theta = prior.sample().to(device)
        theta1, theta2 = theta[0].item(), theta[1].item()
        draw_fake = generate_data(theta1, theta2)[0].to(device)
        mean_, cov_ = encoder(draw_fake)
        mvr_norm = D.MultivariateNormal(mean_, cov_)
        optimizer.zero_grad()
        loss = loss_fcn(theta, mean_, cov_)
        print("Epoch: {}, Loss: {}".format(i, loss.item()))
        loss.backward()
        optimizer.step()

    torch.save(encoder.state_dict(), './trained_encoder.pth')


if __name__ == "__main__":
    main()  