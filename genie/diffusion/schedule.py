import math
import torch


def get_betas(n_timestep, schedule):
	if schedule == 'linear':
		return linear_beta_schedule(n_timestep)
	elif schedule == 'cosine':
		return cosine_beta_schedule(n_timestep)
	else:
		print('Invalid schedule: {}'.format(schedule))
		exit(0)

def linear_beta_schedule(n_timestep, start=0.0001, end=0.02):
    return torch.linspace(start, end, n_timestep)

def cosine_beta_schedule(n_timestep):
    steps = n_timestep + 1
    x = torch.linspace(0, n_timestep, steps)
    alphas_cumprod = torch.cos((x / steps) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)