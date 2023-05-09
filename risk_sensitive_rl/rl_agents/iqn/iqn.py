from functools import partial
import numpy as np
from risk_sensitive_rl.rl_agents.offpolicy import OffPolicyPG

from risk_sensitive_rl.rl_agents.iqn.policy import IQNPolicy

from typing import Optional, Callable

import gym
from risk_sensitive_rl.utils.optimize import optimize, soft_update, build_optimizer

import haiku as hk
from risk_sensitive_rl.rl_agents.risk_models import *


class IQN(OffPolicyPG):
    name = 'SpectralDQN'
    risk_types = {"cvar": sample_cvar,
                  "general_cvar": sample_cvar_general,
                  "general_pow": sample_power_general,
                  "cpw": cpw,
                  "wang": wang,
                  "power": sample_power}

    def __init__(self,
                 env: gym.Env,
                 lr: float = 1e-4,
                 epsilon_greedy_rate: float = 0.01,
                 policy_fn: Optional[Callable] = None,
                 buffer_size: int = 1000_000,
                 gamma: float = 0.99,
                 batch_size: int = 256,
                 warmup_steps: int = 2000,
                 n_quantiles: int = 8,
                 seed: int = 0,
                 wandb_proj: Optional[str] = None,
                 cfg: Optional[dict] = None,
                 work_dir: Optional[str] = None,
                 soft_update_coef: float = 0.05,
                 steps_per_gradients: int = 1,
                 risk_type='cvar',
                 risk_param=1.0,
                 ):
        super().__init__(
            env=env,
            buffer_size=buffer_size,
            gamma=gamma,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            seed=seed,
            steps_per_gradients=steps_per_gradients,
            wandb_proj=wandb_proj,
            work_dir=work_dir,
            cfg=cfg)

        self.n_quantiles = n_quantiles
        self.rng = hk.PRNGSequence(seed)
        if policy_fn is None:
            def q_network(obs, taus):
                return IQNPolicy(actions_dim=self.env.action_space.n, features_dim=256)(obs, taus)

            self.qf = hk.without_apply_rng(hk.transform(q_network))
            obs_placeholder = self.make_placeholder()
            self.param_qf = self.param_qf_target = self.qf.init(next(self.rng),
                                                                obs_placeholder, jnp.ones((1, self.n_quantiles))
                                                                )
            opt_init, self.opt_qf = build_optimizer(lr)
            self.opt_qf_state = opt_init(self.param_qf)

        self.epsilon_greedy_rate = epsilon_greedy_rate
        self.soft_update_coef = soft_update_coef
        self.qf_layer_norms = []

        try:
            self.risk_model = IQN.risk_types[risk_type]
            self.risk_param = risk_param
        except KeyError:
            raise NotImplementedError

        for keys in self.param_qf.keys():
            if 'layer_norm' in keys.split('/')[-1]:
                self.qf_layer_norms.append(keys)

    def make_placeholder(self):
        s = self.env.observation_space.sample()
        return jnp.asarray(s[None])

    @partial(jax.jit, static_argnums=0)
    def layer_norm_update(self, param_qf_target, param_qf):
        # hard update for the layer norm
        for keys in self.qf_layer_norms:
            param_qf_target[keys] = param_qf[keys]
        return param_qf_target

    @partial(jax.jit, static_argnums=0)
    def _predict(self, param_qf, obs, key) -> np.ndarray:
        base_taus = jax.random.uniform(key, shape=(1, self.n_quantiles)).sort(axis=-1)
        taus = self.risk_model(base_taus, self.risk_param)
        qfs = self.qf.apply(
            param_qf, obs, taus
        )
        return qfs.mean(axis=-2).argmax(axis=-1)

    @staticmethod
    def wrap_env(env):
        return env

    def predict(self, observations, state=None, *args, **kwargs) -> int:
        # policy predict and post process to be numpy
        observations = observations[None]
        actions = self._predict(self.param_qf, obs=observations, key=next(self.rng))
        return actions.item()

    def explore(self, observations, state=None, *args, **kwargs) -> int:
        rand = np.random.uniform(0, 1)
        if rand > self.epsilon_greedy_rate:
            predictions = self.predict(observations, state)
        else:
            predictions = np.random.randint(0, self.env.action_space.n, )
        return predictions

    @staticmethod
    @jax.jit
    def quantile_loss(y: jnp.ndarray,
                      x: jnp.ndarray,
                      taus: jnp.ndarray) -> jnp.ndarray:
        pairwise_delta = y[:, None, :] - x[:, :, None]
        abs_pairwise_delta = jnp.abs(pairwise_delta)
        huber = jnp.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
        loss = jnp.abs(taus[..., None] - jax.lax.stop_gradient(pairwise_delta < 0)) * huber
        return loss

    def sample_taus(self, key):
        return self._sample_taus(key, self.n_quantiles)

    @partial(jax.jit, static_argnums=(0,))
    def qf_loss(self,
                param_qf: hk.Params,
                param_qf_target: hk.Params,
                obs: jnp.ndarray,
                actions: jnp.ndarray,
                rewards: jnp.ndarray,
                dones: jnp.ndarray,
                next_obs: jnp.ndarray,
                key: jax.random.PRNGKey
                ):

        key1, key2 = jax.random.split(key, 2)
        _, taus, weight = self.sample_taus(key1)
        _, next_taus, weight = self.sample_taus(key2)
        current_qfs = self.qf.apply(param_qf, obs, taus)
        current_qf = jnp.take_along_axis(current_qfs, actions[..., None], axis=-1).squeeze(axis=-1)
        target_qf = self.compute_target_qf(param_qf_target, param_qf, next_obs, rewards,
                                                     dones, next_taus)

        quantile_loss = (self.quantile_loss(target_qf, current_qf, taus)).mean(axis=-1) * weight
        quantile_loss = quantile_loss.sum(axis=-1).mean()
        return quantile_loss, None

    @partial(jax.jit, static_argnums=(0,))
    def compute_target_qf(self,
                          param_qf_target,
                          param_qf,
                          next_obs,
                          rewards,
                          dones,
                          next_taus,
                          ):

        next_phi = self.risk_model(next_taus, self.risk_param)
        next_phi_qf = self.qf.apply(param_qf, next_obs, next_phi)
        next_actions = next_phi_qf.mean(axis=-2).argmax(axis=-1, keepdims=True)
        next_qf_quantiles = self.qf.apply(param_qf_target, next_obs, next_taus)

        # (batch, n_taus, n_actions)  -> (batch, n_taus, 1) -> (batch, n_taus)
        next_qf_quantiles = jnp.take_along_axis(next_qf_quantiles, next_actions[..., None], axis=-1).squeeze(axis=-1)
        target_quantile = rewards + (1. - dones) * self.gamma * next_qf_quantiles.sort(axis=-1)

        return jax.lax.stop_gradient(target_quantile)

    def train_step(self):
        obs, actions, rewards, dones, next_obs = self.buffer.sample(self.batch_size)
        self.opt_qf_state, self.param_qf, qf_loss, _ = optimize(
            self.qf_loss,
            self.opt_qf,
            self.opt_qf_state,
            self.param_qf,
            self.param_qf_target,
            obs, actions, rewards, dones, next_obs, next(self.rng))

        self.param_qf_target = soft_update(self.param_qf_target, self.param_qf, self.soft_update_coef)
        self.param_qf_target = self.layer_norm_update(self.param_qf_target, self.param_qf)
        self.logger.record(key='train/quantile_loss', value=qf_loss.item())

    def save(self, path):
        params = {"param_qf": self.param_qf}
        return np.savez(path, **params)

    def load(self, path):
        params = np.load(path, allow_pickle=True)
        self.param_qf = params["param_qf"].item()