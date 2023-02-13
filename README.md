# Genie: De Novo Protein Design by Equivariantly Diffusing Oriented Residue Clouds

This repository provides the implementation code for our [preprint](https://arxiv.org/abs/2301.12485).

## Installation

Clone this repository and go into the root directory. Set up the package by running `pip install -e .`. This would automatically install dependencies needed for the code, including logging packages like tensorboard and wandb.

### Data Download
We provide scripts that we use for downloading and cleaning SCOPe dataset. To download, run
```
chmod +x scripts/install_dataset.sh
./scripts/install_dataset.sh
```

## Training

To train Genie, create a directory `runs/[RUN_NAME]` and go into the directory. Create a configuration file with name `configuration`. An example of configuration file is provided in `example_configuration` and a complete list of configurable parameters could be found in `genie/config.py`. Note that in the configuration file, `name` should match with `RUN_NAME` in order to log into the correct directory. To start training, run
```
python genie/train.py -c runs/RUN_NAME/configuration -g0 &
```
for example, to run in the background on GPU 0.

## Sampling

To sample domains using your own trained Genie, run
```
python genie/sample.py -n RUN_NAME -g0
```
By default, it uses the checkpoint with the latest version and epoch. You could also specify the version and epoch by using the `-v` and `-e` flag respectively. This would sample 10 domains per sequence length between 50 and 128, with a sampling batch size of 5. The output are stored in the directory `runs/[RUN_NAME]/version_[VERSION]/samples/epoch_[EPOCH]`.

We also provide the weights for our trained model, which are available under the `weights` directory, together with the corresponding configuration file. To load the model, run
```
from genie.config import Config
from genie.diffusion.genie import Genie

config = Config('weights/configuration')
model = Genie.load_from_checkpoint('weights/genie_l_128_epoch=49999.ckpt', config=config)
```
