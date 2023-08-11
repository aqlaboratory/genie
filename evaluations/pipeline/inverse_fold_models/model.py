from abc import ABC, abstractmethod

class InverseFoldModel(ABC):

	@abstractmethod
	def predict(self, pdb_filepath):
		raise NotImplemented