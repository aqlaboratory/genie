import os
import sys
import glob
import numpy as np

from genie.config import Config
from genie.diffusion.genie import Genie


def get_versions(rootdir, name):
	basedir = os.path.join(rootdir, name)
	return sorted([
		int(version_dir.split('_')[-1]) 
		for version_dir in glob.glob(os.path.join(basedir, 'version_*'), recursive=False)
	])

def get_epochs(rootdir, name, version):
	basedir = os.path.join(rootdir, name)
	return sorted([
		int(epoch_filepath.split('=')[-1].split('.')[0])
		for epoch_filepath in glob.glob(os.path.join(basedir, 'version_{}'.format(version), 'checkpoints', '*.ckpt'))
	])

def load_model(rootdir, name, version=None, epoch=None):

	# load configuration and create default model
	basedir = os.path.join(rootdir, name)
	config_filepath = os.path.join(basedir, 'configuration')
	config = Config(config_filepath)

	# check for latest version if needed
	available_versions = get_versions(rootdir, name)
	if version is None:
		if len(available_versions) == 0:
			print('No checkpoint available (version)')
			sys.exit(0)
		version = np.max(available_versions)
	else:
		if version not in available_versions:
			print('Missing checkpoint version: {}'.format(version))
			sys.exit(0)

	# check for latest epoch if needed
	available_epochs = get_epochs(rootdir, name, version)
	if epoch is None:
		if len(available_epochs) == 0:
			print('No checkpoint available (epoch)')
			sys.exit(0)
		epoch = np.max(available_epochs)
	else:
		if epoch not in available_epochs:
			print('Missing checkpoint epoch: {}'.format(epoch))
			sys.exit(0)

	# load checkpoint
	ckpt_filepath = os.path.join(basedir, 'version_{}'.format(version), 'checkpoints', 'epoch={}.ckpt'.format(epoch))
	diffusion = Genie.load_from_checkpoint(ckpt_filepath, config=config)
	
	# save checkpoint information
	diffusion.rootdir = rootdir
	diffusion.name = name
	diffusion.version = version
	diffusion.epoch = epoch
	diffusion.checkpoint = ckpt_filepath

	return diffusion
