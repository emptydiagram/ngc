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
