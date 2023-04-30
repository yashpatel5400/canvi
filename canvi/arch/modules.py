import torch
import torch.distributions as D
import torch.nn as nn

class TransformedUniform(nn.Module):
    '''
    A Uniform Random variable defined on the real line.
    Transformed by sigmoid to reside between low and high.
    '''
    def __init__(self, low=0., high=1.):
        super().__init__()
        self.low = low
        self.high = high
        self.length = high-low
        self.instance = D.Uniform(low=self.low, high=self.high)
        self.jitter = 1e-8

    def transform(self, value: torch.Tensor):
        tt = torch.sigmoid(value)*self.length+self.low
        clamped = tt.clamp(min=self.low+self.jitter, max=self.high-self.jitter)
        return clamped
    
    def inv_transform(self, tval: torch.Tensor):
        assert (self.low <= tval).all(), "Input is outside of the support."
        assert (self.high >= tval).all(), "Input is outside of the support."
        to_invert = (tval-self.low)/self.length
        return torch.logit(to_invert)
    
    def log_prob(self, value: torch.Tensor):
        tval = self.transform(value)
        return self.instance.log_prob(tval)
    
    def sample(self, shape):
        tsamples = self.instance.sample(shape)
        return self.inv_transform(tsamples)