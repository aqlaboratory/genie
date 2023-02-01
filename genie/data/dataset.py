import numpy as np
from torch.utils.data import Dataset

from genie.utils.data_io import load_coord


class SCOPeDataset(Dataset):
	# Assumption: all domains have at least n_res residues

	def __init__(self, filepaths, max_n_res, min_n_res):
		super(SCOPeDataset, self).__init__()
		self.filepaths = filepaths
		self.max_n_res = max_n_res
		self.min_n_res = min_n_res

	def __len__(self):
		return len(self.filepaths)

	def __getitem__(self, idx):
		coords = load_coord(self.filepaths[idx])
		n_res = int(len(coords) / 3)
		if self.max_n_res is not None:
			coords = np.concatenate([coords, np.zeros(((self.max_n_res - n_res) * 3, 3))], axis=0)
			mask = np.concatenate([np.ones(n_res), np.zeros(self.max_n_res - n_res)])
		else:
			assert self.min_n_res is not None
			s_idx = np.random.randint(n_res - self.min_n_res + 1)
			start_idx = s_idx * 3
			end_idx = (s_idx + self.min_n_res) * 3
			coords = coords[start_idx:end_idx]
			mask = np.ones(self.min_n_res)
		return coords, mask