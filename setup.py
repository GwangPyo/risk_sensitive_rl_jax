from setuptools import setup, find_packages

setup(
    name='risk_rl_jax',
    version='0.1',
    description='risk sensitive RL packages for continuous control',
    author='Gwangpyo Yoo',
    author_email='necrocathy@gmail.com',
    requires=['jax', 'dm-haiku', 'optax', 'gym==0.21', 'numpy'],
    packages=find_packages(exclude=[])
)
