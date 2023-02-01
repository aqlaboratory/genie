import torch
import numpy as np


def distance(p, eps=1e-10):
    # [*, 2, 3]
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5

def dihedral(p, eps=1e-10):
    # p: [*, 4, 3]

    # [*, 3]
    u1 = p[..., 1, :] - p[..., 0, :]
    u2 = p[..., 2, :] - p[..., 1, :]
    u3 = p[..., 3, :] - p[..., 2, :]

    # [*, 3]
    u1xu2 = torch.cross(u1, u2, dim=-1)
    u2xu3 = torch.cross(u2, u3, dim=-1)

    # [*]
    u2_norm = (eps + torch.sum(u2 ** 2, dim=-1)) ** 0.5
    u1xu2_norm = (eps + torch.sum(u1xu2 ** 2, dim=-1)) ** 0.5
    u2xu3_norm = (eps + torch.sum(u2xu3 ** 2, dim=-1)) ** 0.5

    # [*]
    cos_enc = torch.einsum('...d,...d->...', u1xu2, u2xu3)/ (u1xu2_norm * u2xu3_norm)
    sin_enc = torch.einsum('...d,...d->...', u2, torch.cross(u1xu2, u2xu3, dim=-1)) /  (u2_norm * u1xu2_norm * u2xu3_norm)

    return torch.stack([cos_enc, sin_enc], dim=-1)


def compute_frenet_frames(x, mask, eps=1e-10):
    # x: [b, n_res, 3]

    t = x[:, 1:] - x[:, :-1]
    t_norm = torch.sqrt(eps + torch.sum(t ** 2, dim=-1))
    t = t / t_norm.unsqueeze(-1)

    b = torch.cross(t[:, :-1], t[:, 1:])
    b_norm = torch.sqrt(eps + torch.sum(b ** 2, dim=-1))
    b = b / b_norm.unsqueeze(-1)

    n = torch.cross(b, t[:, 1:])

    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    # TODO: recheck correctness of this implementation
    rots = []
    for i in range(mask.shape[0]):
        rots_ = torch.eye(3).unsqueeze(0).repeat(mask.shape[1], 1, 1)
        length = torch.sum(mask[i]).int()
        rots_[1:length-1] = tbn[i, :length-2]
        rots_[0] = rots_[1]
        rots_[length-1] = rots_[length-2]
        rots.append(rots_)
    rots = torch.stack(rots, dim=0).to(x.device)

    return rots