import wandb
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from genie.config import Config
from genie.data.data_module import SCOPeDataModule
from genie.diffusion.genie import Genie


def main(args):

	# configuration
	config = Config(filename=args.config)

	# devices
	gpus = [int(elt) for elt in args.gpus.split(',')] if args.gpus is not None else None

	# logger
	tb_logger = TensorBoardLogger(
		save_dir=config.io['log_dir'],
		name=config.io['name']
	)
	wandb_logger = WandbLogger(project=config.io['name'])

	# checkpoint callback
	checkpoint_callback = ModelCheckpoint(
		every_n_epochs=config.training['checkpoint_every_n_epoch'],
		filename='{epoch}',
		save_top_k=-1
	)

	# seed
	seed_everything(config.training['seed'], workers=True)

	# data module
	dm = SCOPeDataModule(**config.io, batch_size=config.training['batch_size'])

	# model
	model = Genie(config)

	# trainer
	trainer = Trainer(
		gpus=gpus,
		logger=[tb_logger, wandb_logger],
		strategy='ddp',
		deterministic=True,
		enable_progress_bar=False,
		log_every_n_steps=config.training['log_every_n_step'],
		max_epochs=config.training['n_epoch'],
		callbacks=[checkpoint_callback]
	)

	# run
	trainer.fit(model, dm)


if __name__ == '__main__':

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpus', type=str, help='GPU devices to use')
	parser.add_argument('-c', '--config', type=str, help='Path for configuration file', required=True)
	args = parser.parse_args()

	# run
	main(args)