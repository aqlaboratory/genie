import esm
import torch
import numpy as np

from fold_models.model import FoldModel


class ESMFold(FoldModel):

	def __init__(self):
		self.model = esm.pretrained.esmfold_v1()
		self.model = self.model.eval().cuda()

	def predict(self, seq):
		with torch.no_grad():
			output = self.model.infer(seq, num_recycles=3)
			pdb_str = self.model.output_to_pdb(output)[0]
			pae = (output['aligned_confidence_probs'].cpu().numpy()[0] * np.arange(64)).mean(-1) * 31
			mask = output['atom37_atom_exists'].cpu().numpy()[0,:,1] == 1
		return pdb_str, pae[mask,:][:,mask]
