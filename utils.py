import random

import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_gaussian_dense(dims, stddev, device):
    return torch.empty(dims, requires_grad=False, device=device).normal_(mean=0.0, std=stddev)

def make_moving_collate_fn(device):
    def moving_collate(batch):
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs).to(device)
        targets = torch.tensor(targets).to(device)
        return inputs, targets

    return moving_collate

# make the lateral K-WTA matrix
def make_lkwta(dim, K, inh_coeff, exc_coeff):
    assert dim % K == 0, "dim must be divisible by K"

    I = torch.eye(dim, requires_grad=False)
    M = torch.zeros(dim, dim, requires_grad=False)
    # builds a dim x dim block diagonal matrix with K x K identity blocks
    for i in range(dim // K):
        M[i*K:(i+1)*K, i*K:(i+1)*K] = 1.0
    return inh_coeff * M * (1 - I) - exc_coeff * I
