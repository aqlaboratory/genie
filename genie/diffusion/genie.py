import torch

from genie.diffusion.diffusion import Diffusion
from genie.diffusion.schedule import get_betas
from genie.utils.loss import rmsd
from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames


class Genie(Diffusion):

	def setup_schedule(self):

		self.betas = get_betas(self.config.diffusion['n_timestep'], self.config.diffusion['schedule']).to(self.device)
		self.alphas = 1. - self.betas
		self.alphas_cumprod = torch.cumprod(self.alphas, 0)
		self.alphas_cumprod_prev = torch.cat([torch.Tensor([1.]).to(self.device), self.alphas_cumprod[:-1]])
		self.one_minus_alphas_cumprod = 1. - self.alphas_cumprod
		self.one_minus_alphas_cumprod_prev = 1. - self.alphas_cumprod_prev
		
		self.sqrt_betas = torch.sqrt(self.betas)
		self.sqrt_alphas = torch.sqrt(self.alphas)
		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = 1. / self.sqrt_alphas_cumprod
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

		self.posterior_mean_coef1 = self.betas * self.sqrt_alphas_cumprod_prev / self.one_minus_alphas_cumprod
		self.posterior_mean_coef2 = self.one_minus_alphas_cumprod_prev * self.sqrt_alphas / self.one_minus_alphas_cumprod
		self.posterior_variance = self.betas * self.one_minus_alphas_cumprod_prev / self.one_minus_alphas_cumprod

	def transform(self, batch):

		coords, mask = batch
		coords = coords.float()
		mask = mask.float()

		ca_coords = coords[:, 1::3]
		trans = ca_coords - torch.mean(ca_coords, dim=1, keepdim=True)
		rots = compute_frenet_frames(trans, mask)

		return T(rots, trans), mask

	def sample_timesteps(self, num_samples):
		return torch.randint(0, self.config.diffusion['n_timestep'], size=(num_samples,)).to(self.device)

	def sample_frames(self, mask):
		trans = torch.randn((mask.shape[0], mask.shape[1], 3)).to(self.device)
		trans = trans * mask.unsqueeze(-1)
		rots = compute_frenet_frames(trans, mask)
		return T(rots, trans)

	def q(self, t0, s, mask):

		# [b, n_res, 3]
		trans_noise = torch.randn_like(t0.trans) * mask.unsqueeze(-1)
		rots_noise = torch.eye(3).view(1, 1, 3, 3).repeat(t0.shape[0], t0.shape[1], 1, 1).to(self.device)

		trans = self.sqrt_alphas_cumprod[s].view(-1, 1, 1).to(self.device) * t0.trans + \
			self.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1).to(self.device) * trans_noise
		rots = compute_frenet_frames(trans, mask)

		return T(rots, trans), T(rots_noise, trans_noise)

	def p(self, ts, s, mask):

		# [b, 1, 1]
		w_noise = ((1. - self.alphas[s].to(self.device)) / self.sqrt_one_minus_alphas_cumprod[s].to(self.device)).view(-1, 1, 1)

		# [b, n_res]
		noise_pred_trans = ts.trans - self.model(ts, s, mask).trans
		noise_pred_rots = torch.eye(3).view(1, 1, 3, 3).repeat(ts.shape[0], ts.shape[1], 1, 1)
		noise_pred = T(noise_pred_rots, noise_pred_trans)

		# [b, n_res, 3]
		trans_mean = (1. / self.sqrt_alphas[s]).view(-1, 1, 1).to(self.device) * (ts.trans - w_noise * noise_pred.trans)
		trans_mean = trans_mean * mask.unsqueeze(-1)

		if (s == 0.0).all():
			rots_mean = compute_frenet_frames(trans_mean, mask)
			return T(rots_mean.detach(), trans_mean.detach())
		else:

			# [b, n_res, 3]
			trans_z = torch.randn_like(ts.trans).to(self.device)

			# [b, 1, 1]
			trans_sigma = self.sqrt_betas[s].view(-1, 1, 1).to(self.device)

			# [b, n_res, 3]
			trans = trans_mean + trans_sigma * trans_z
			trans = trans * mask.unsqueeze(-1)

			# [b, n_res, 3, 3]
			rots = compute_frenet_frames(trans, mask)

			return T(rots.detach(), trans.detach())

	def loss_fn(self, tnoise, ts, s, mask):

		noise_pred_trans = ts.trans - self.model(ts, s, mask).trans

		trans_loss = rmsd(
			noise_pred_trans,
			tnoise.trans,
			mask
		)

		return trans_loss