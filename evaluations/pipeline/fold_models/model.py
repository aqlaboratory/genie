from abc import ABC, abstractmethod

class FoldModel(ABC):

	@abstractmethod
	def predict(self, seq):
		raise NotImplemented