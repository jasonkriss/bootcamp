import torch
import numpy as np
from torch.autograd import Variable

def compute_return(rewards, gamma=1.0):
    discounts = gamma ** np.arange(rewards.shape[0])
    return np.sum(discounts * rewards)

def make_tensor(array, dtype='float32'):
    return torch.from_numpy(array.astype(dtype))

def make_variable(array, dtype='float32'):
    return Variable(make_tensor(array, dtype))
