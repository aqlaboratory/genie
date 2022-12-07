from setuptools import setup

setup(
      name='psddpm',
      version='0.0.1',
      description='Protein Structure Denoising Diffusion Probabilistic Model',
      packages=['psddpm'],
      install_requires=[
            'tqdm',
            'numpy',
            'torch',
            'scipy',
            'wandb',
            'pytorch_lightning',
      ],
)