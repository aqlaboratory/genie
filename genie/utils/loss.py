import torch


def rmsd(x_pred, x, mask, eps=1e-10):
    rmsds = (eps + torch.sum((x_pred - x) ** 2, dim=-1)) ** 0.5
    return torch.sum(rmsds * mask) / torch.sum(mask)