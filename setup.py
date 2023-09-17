from setuptools import setup

setup(
      name='genie',
      version='0.0.1',
      description='de novo protein design through equivariantly diffusing oriented residue clouds',
      packages=['genie'],
      install_requires=[
            'urllib3==1.26.14',
            'charset-normalizer==2.1.1',
            'tqdm',
            'numpy',
            'torch',
            'scipy',
            'wandb',
            'pandas',
            'tensorboard',
            'pytorch_lightning',
      ],
)