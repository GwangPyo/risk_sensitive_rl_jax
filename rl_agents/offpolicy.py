from utils.replay_buffer import ReplayBuffer
import gym
from abc import ABCMeta, abstractmethod
import numpy as np
import jax.numpy as jnp
import haiku as hk

from time import time
from datetime import timedelta
from collections import deque
from utils.logger import Logger
from utils.env_wrappers import NormalizedActionWrapper
from functools import partial
import jax
import os


class OffPolicyPG(object, metaclass=ABCMeta):
    name = None

    def __init__(self,
                 env: gym.Env,
                 buffer_size: int = 1000_000,
                 gamma: float = 0.99,
                 batch_size: int = 256,
                 warmup_steps: int = 2000,
                 seed: int = 0,
                 wandb: bool = False,
                 steps_per_gradients: int = 1,
                 ):
        """
        :param env: environment to learn
        :param buffer_size: size of replay buffer
        :param gamma: discount factor 0 <= gamma <= 1
        :param batch_size: batch size to train
        :param warmup_steps: data collection steps without train
        :param seed: random seed number
        :param wandb: use wandb load or not
        """
        env.seed(seed)
        self.env = self.wrap_env(env)
        self.seed = seed
        self._last_obs = env.reset()
        assert isinstance(env.action_space, gym.spaces.Box)
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.buffer = self.make_buffer(buffer_size)

        np.random.seed(seed)
        self.batch_size = batch_size
        self.keygen = hk.PRNGSequence(seed)
        self._last_scores = 0
        self._last_scores_100 = deque(maxlen=100)
        self._epi_len = 0
        self._epi_len_100 = deque(maxlen=100)
        self._successes = deque(maxlen=100)
        self._n_episodes = 0
        self._n_updates = 0
        self._log_interval = 1
        self._steps_per_gradients = steps_per_gradients

        if not wandb:
            self.logger = Logger()
        else:
            self.logger = Logger(
                "log_{}".format(seed),
                output_formats=['stdout', 'wandb'],
                fmt_args=(self.name, )
            )

    @staticmethod
    def wrap_env(env):
        return NormalizedActionWrapper(env)

    def make_buffer(self, buffer_size):
        return ReplayBuffer(buffer_size=buffer_size,
                            observation_space=self.env.observation_space,
                            action_space=self.env.action_space)

    def make_placeholder(self):
        a = self.env.action_space.sample()
        s = self.env.observation_space.sample()
        return jnp.asarray(s[None]), jnp.asarray(a[None])

    @abstractmethod
    def predict(self, observations, state=None, *args, **kwargs) -> np.ndarray:
        # policy predict and post process to be numpy
        pass

    def explore(self, observations, state=None, *args, **kwargs) -> np.ndarray:
        return self.predict(observations, state, *args, **kwargs)

    def collect_sample(self, warmup=False):
        if not warmup:
            action = self.explore(self._last_obs)
        else:
            action = self.env.action_space.sample()
        next_observation, reward, done, info = self.env.step(action)
        self._last_scores += reward
        self._epi_len += 1
        self.buffer.append(self._last_obs, action, reward, done, next_observation)

        if done:
            self._last_obs = self.env.reset().copy()
            if 'is_success' in info.keys():
                self._successes.append(float(info['is_success']))
                self.logger.record('rollout/success ratio', str(np.round(100 * np.mean(self._successes))) + '%')

            self.logger.record('rollout/return', self._last_scores)
            self._last_scores_100.append(self._last_scores)
            self.logger.record('rollout/last_100_return', np.mean(self._last_scores_100))

            self.logger.record('rollout/epi_len', self._epi_len)
            self._epi_len_100.append(self._epi_len)
            self._n_episodes += 1
            self.logger.record('rollout/last_100epi_len', np.mean(self._epi_len_100))
            self.logger.record('rollout/n_episodes', self._n_episodes)
            self.logger.record('time/n_updates', self._n_updates)
            self._last_scores = 0
            self._epi_len = 0
            self.done_callback()
            if self._n_episodes % self._log_interval == 0:
                self.logger.dump()
        else:
            self._last_obs = next_observation.copy()

    @abstractmethod
    def train_step(self):
        pass

    def done_callback(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def sample_taus(self, key, placeholder):
        presume_tau = jax.random.uniform(key, placeholder.shape) + 0.1
        presume_tau = presume_tau / presume_tau.sum(axis=-1, keepdims=True)
        tau = jnp.cumsum(presume_tau, axis=-1)
        tau_hat = jnp.zeros_like(tau)
        tau_hat = tau_hat.at[:, 0:1].set(tau[:, 0:1] / 2)
        tau_hat = tau_hat.at[:, 1:].set( (tau[:, 1:] + tau[:, :-1])/2)
        return jax.lax.stop_gradient(tau), jax.lax.stop_gradient(tau_hat), jax.lax.stop_gradient(tau_hat)

    def learn(self,
              steps,
              log_interval=1,
              auto_save_interval=None,
              auto_save_kwargs=None):

        start_time = time()
        self._log_interval = log_interval

        for _ in range(self.warmup_steps):
            self.collect_sample(warmup=True)

        for step in range(self.warmup_steps, steps):

            self.collect_sample(warmup=False)

            if step % self._steps_per_gradients == 0:
                update_start = time()
                self.train_step()
                self._n_updates += 1
                update_end = time()
                fps = 1/(update_end - update_start)

                self.logger.record('time/elapsed', timedelta(seconds=int(update_end - start_time)))
            remaining_step = (steps - step)

            eta = (remaining_step / fps)
            self.logger.record_mean('time/fps', fps, exclude='tensorboard')
            self.logger.record('time/eta', timedelta(seconds=int(eta)))
            self.logger.record('time/step', step)
            self.logger.record('time/progress', str(round(100 * step/steps, 2)) + "%")

            if auto_save_interval and (step % auto_save_interval == 0):
                self.auto_save(step=step, **auto_save_kwargs)
        if auto_save_interval:
            self.auto_save(step=step, **auto_save_kwargs)

    def auto_save(self, step, name=None, dir=None):
        if name is None:
            name = self.name
        if dir is not None and not(os.path.exists(dir)):
            os.makedirs(dir)
        if dir is not None:
            save_format = "{}/{}_step_{}_{}".format(dir, step, name, self.seed)
        else:
            save_format = "{}_step_{}_{}".format(step, name, self.seed)
        return self.save(save_format)

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

