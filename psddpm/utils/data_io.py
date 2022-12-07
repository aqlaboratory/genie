import os
import glob
import torch
import numpy as np


def load_coord(filepath):
	return np.loadtxt(filepath, delimiter=',')

def load_classes(filepath):
	classes = {}
	with open(filepath) as file:
		for line in file:
			elts = line.strip().split(',')
			classes[elts[0]] = elts[1].split('.')[0]
	return classes

def load_filepaths(datadir, dataset_names, max_n_res=None, min_n_res=None, classes=None, n_data=None):

	# load
	output_filepaths = []
	if dataset_names is None:
		print('Error: Missing datasets')
		exit(0)
	elif 'scope' in dataset_names or 'cath' in dataset_names:
		if len(dataset_names) > 1:
			print('Error: Incompatible dataset')
			exit(0)
		rootdir = os.path.join(datadir, dataset_names[0])
		output_filepaths.extend(glob.glob(os.path.join(rootdir, 'coords', '*.npy')))
	else:
		if classes is not None:
			print('Error: classes not found')
			exit(0)
		for dataset_name in dataset_names:
			output_filepaths.extend(glob.glob(os.path.join(datadir, f'{dataset_name}_coords', '*.npy')))

	# filter by number of residues
	min_n_res = 0 if min_n_res is None else min_n_res
	max_n_res = 1e10 if max_n_res is None else max_n_res
	filepaths, output_filepaths = output_filepaths, []
	for filepath in filepaths:
		coords = load_coord(filepath)
		n_res = int(coords.shape[0] / 3)
		if n_res >= min_n_res and n_res <= max_n_res:
			output_filepaths.append(filepath)

	# sample
	if classes is None:
		if n_data is not None:
			assert n_data <= len(output_filepaths)
			output_filepaths = output_filepaths[:n_data]
		return output_filepaths

	# compute samples per class
	samples_per_class = None
	if n_data is not None:
		samples_per_class = int(n_data / len(classes))
		assert samples_per_class == int(n_data / len(classes))

	# separate by classes
	filepaths_by_class = {}
	domain_classes = load_classes(os.path.join(rootdir, 'classes.txt'))
	for filepath in output_filepaths:
		domain_name = filepath.split('/')[-1].split('.')[0]
		domain_class = domain_classes[domain_name]
		if domain_class not in filepaths_by_class:
			filepaths_by_class[domain_class] = []
		filepaths_by_class[domain_class].append(filepath)

	# filter by classes
	output_filepaths = []
	for selected_class in classes:
		assert selected_class in filepaths_by_class
		filepaths = filepaths_by_class[selected_class]
		if samples_per_class is not None:
			assert samples_per_class <= len(filepaths)
			filepaths = filepaths[:samples_per_class]
		output_filepaths.extend(filepaths)

	return output_filepaths
