import argparse

from pipeline import Pipeline
from inverse_fold_models.proteinmpnn import ProteinMPNN
from fold_models.esmfold import ESMFold

def main(args):

	# inverse fold model
	inverse_fold_model = ProteinMPNN()

	# fold model
	fold_model = ESMFold()

	# pipeline
	pipeline = Pipeline(inverse_fold_model, fold_model)

	# additional information
	info = {}
	if args.motif_filepath:
		info['motif_filepath'] = args.motif_filepath

	# evaluate
	pipeline.evaluate(args.input_dir, args.output_dir, info=info)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, help='Input directory', required=True)
	parser.add_argument('--output_dir', type=str, help='Output directory', required=True)
	parser.add_argument('--motif_filepath', type=str, help='Motif filepath (for motif scaffolding evaluation)')
	args = parser.parse_args()
	main(args)