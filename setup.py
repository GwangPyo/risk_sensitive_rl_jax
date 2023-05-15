from setuptools import setup, find_packages

setup(
    name='risk_rl_jax',
    version='0.1.1',
    description='risk sensitive RL packages for continuous control',
    author='Gwangpyo Yoo',
    author_email='necrocathy@gmail.com',
    requires=[],
    install_requires=['wandb', 'dm-haiku', 'optax', 'gym==0.21', 'fire'],
    packages=find_packages(exclude=[])
)
