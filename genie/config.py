import os

int_or_none = lambda x: int(x) if x is not None else None
str_list_or_none = lambda x: x.strip().split(',') if x is not None else None
int_list_or_none = lambda x: int(x.strip().split(',')) if x is not None else None
eval_if_str = lambda x: literal_eval(x) if isinstance(x, str) else x

class Config:

	def __init__(self, filename=None):
		config = {} if filename is None else self._load_config(filename)
		self._create_config(config)

	def _create_config(self, config):

		self.io = {
			'name':                             config.get('name',               None),
			'max_n_res':            int_or_none(config.get('maximumNumResidues', None)),
			'min_n_res':            int_or_none(config.get('minimumNumResidues', None)),
			'log_dir':                          config.get('logDirectory',       'runs'),
			'data_dir':                         config.get('dataDirectory',      'data'),
			'dataset_names':   str_list_or_none(config.get('datasetNames',       'scope')),
			'dataset_size':         int_or_none(config.get('datasetSize',        None)),
			'dataset_classes': str_list_or_none(config.get('datasetClasses',     None))
		}

		self.diffusion = {
			'n_timestep': int(config.get('numTimesteps', 1000)),
			'schedule':       config.get('schedule',     'cosine'),
		}

		self.model = {

			# general
			'c_s':                            int(config.get('singleFeatureDimension',                  128)),
			'c_p':                            int(config.get('pairFeatureDimension',                    128)),

			# single feature network
			'c_pos_emb':                      int(config.get('positionalEmbeddingDimension',            128)),
			'c_timestep_emb':                 int(config.get('timestepEmbeddingDimension',              128)),

			# pair feature network
			'relpos_k':                       int(config.get('relativePositionK',                       32)),
			'template_type':                      config.get('templateType',                            'v1'),

			# pair transform network
			'n_pair_transform_layer':         int(config.get('numPairTransformLayers',                  5)),
			'include_mul_update':                 config.get('includeTriangularMultiplicativeUpdate',   True),
			'include_tri_att':                    config.get('includeTriangularAttention',              False),
			'c_hidden_mul':                   int(config.get('triangularMultiplicativeHiddenDimension', 128)),
			'c_hidden_tri_att':               int(config.get('triangularAttentionHiddenDimension',      32)),
			'n_head_tri':                     int(config.get('triangularAttentionNumHeads',             4)),
			'tri_dropout':                  float(config.get('triangularDropout',                       0.25)),
			'pair_transition_n':              int(config.get('pairTransitionN',                         4)),

			# structure network
			'n_structure_layer':              int(config.get('numStructureLayers',                      5)),
			'n_structure_block':              int(config.get('numStructureBlocks',                      1)),
			'c_hidden_ipa':                   int(config.get('ipaHiddenDimension',                      16)),
			'n_head_ipa':                     int(config.get('ipaNumHeads',                             12)),
			'n_qk_point':                     int(config.get('ipaNumQkPoints',                          4)),
			'n_v_point':                      int(config.get('ipaNumVPoints',                           8)),
			'ipa_dropout':                  float(config.get('ipaDropout',                              0.1)),
			'n_structure_transition_layer':   int(config.get('numStructureTransitionLayers',            1)),
			'structure_transition_dropout': float(config.get('structureTransitionDropout',              0.1))

		}

		self.training = {
			'seed':                     int(config.get('seed',                   100)),
			'n_epoch':                  int(config.get('numEpoches',             1)),
			'batch_size':               int(config.get('batchSize',              32)),
			'log_every_n_step':         int(config.get('logEverySteps',          1000)),
			'checkpoint_every_n_epoch': int(config.get('checkpointEveryEpoches', 500)),
		}

		self.optimization = {
			'lr': float(config.get('learningRate', 1e-4))
		}

	def _load_config(self, filename):
		config = {}
		with open(filename) as file:
			for line in file:
				elts = line.split()
				if len(elts) == 2:
					if elts[1] == 'True':
						config[elts[0]] = True
					elif elts[1] == 'False':
						config[elts[0]] = False
					else:
						config[elts[0]] = elts[1]
		return config