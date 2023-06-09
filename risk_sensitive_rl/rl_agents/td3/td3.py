from functools import partial
import gym

from risk_sensitive_rl.rl_agents.offpolicy import OffPolicyPG
from risk_sensitive_rl.utils.optimize import optimize, build_optimizer, soft_update
from risk_sensitive_rl.rl_agents.td3.policy import DeterministicActor, Critic
from risk_sensitive_rl.rl_agents.risk_models import *

import numpy as np
import haiku as hk

from typing import Callable, Optional


class TD3(OffPolicyPG):
    name = "TD3"
    risk_types = {"cvar": sample_cvar,
                  "general_cvar": sample_cvar_general,
                  "general_pow": sample_power_general,
                  "cpw": cpw,
                  "wang": wang,
                  "power": sample_power}

    def __init__(self,
                 env: gym.Env,
                 buffer_size: int = 1000_000,
                 gamma: float = 0.99,
                 batch_size: int = 128,
                 warmup_steps: int = 2000,
                 seed: int = 0,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 delay: int = 2,
                 soft_update_coef: float = 5e-3,
                 target_noise: float = 0.3,
                 target_noise_clip: float = 0.5,
                 drop_per_net: int = 5,
                 risk_type: str = 'cvar',
                 risk_param: float = 1.,
                 wandb_proj: Optional[str] = None,
                 cfg: Optional[dict] = None,
                 work_dir: Optional[str] = None,
                 actor_fn: Callable = None,
                 critic_fn: Callable = None,
                 explore_noise: float = 0.3,
                 exploration_noise_clip: float = 0.5,
                 n_critics: int = 2,
                 ):
        try:
            self.risk_model = TD3.risk_types[risk_type]
            self.risk_param = risk_param
        except KeyError:
            raise NotImplementedError

        self.risk_name = risk_type
        self.rng = hk.PRNGSequence(seed)
        super().__init__(env,
                         buffer_size=buffer_size,
                         gamma=gamma,
                         batch_size=batch_size,
                         warmup_steps=warmup_steps,
                         seed=seed,
                         wandb_proj=wandb_proj,
                         work_dir=work_dir,
                         cfg=cfg)
        n_quantiles = 32
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics

        if actor_fn is None:
            def actor_fn(obs):
                return DeterministicActor(self.env.action_space.shape[0])(obs)

            obs_placeholder, a_placeholder = self.make_placeholder()
            self.actor = hk.without_apply_rng(hk.transform(actor_fn))
            self.param_actor = self.param_actor_target = self.actor.init(next(self.rng), obs_placeholder)

        if critic_fn is None:
            def critic_fn(obs, actions, taus):
                return Critic(n_critics=self.n_critics)(obs, actions, taus)

            obs_placeholder, a_placeholder = self.make_placeholder()
            quantile_placeholder = jnp.ones((1, self.n_quantiles))
            self.critic = hk.without_apply_rng(hk.transform(critic_fn))
            self.param_critic = self.param_critic_target = self.critic.init(next(self.rng),
                                                                            obs_placeholder,
                                                                            a_placeholder,
                                                                            quantile_placeholder)

        opt_init, self.opt_actor = build_optimizer(lr_actor)
        self.opt_actor_state = opt_init(self.param_actor)

        opt_init, self.opt_critic = build_optimizer(lr_critic)
        self.opt_critic_state = opt_init(self.param_critic)

        self._n_updates = 0

        self.explore_noise = explore_noise
        self.exploration_noise_clip = exploration_noise_clip
        self.drop_per_net = drop_per_net
        self.delay = delay
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.soft_update_coef = soft_update_coef
        self.taus_placeholder = jnp.ones((self.batch_size, self.n_quantiles), dtype=jnp.float32)

        self.critic_layer_norms = []
        self.actor_layer_norms = []
        for keys in self.param_critic.keys():
            if 'layer_norm' in keys.split('/')[-1]:
                self.critic_layer_norms.append(keys)

        for keys in self.param_actor.keys():
            if 'layer_norm' in keys.split('/')[-1]:
                self.critic_layer_norms.append(keys)

    @partial(jax.jit, static_argnums=0)
    def layer_norm_update(self, param_critic_target, param_critic, param_actor_target, param_actor):
        # hard update for the layer norm
        for key in self.critic_layer_norms:
            param_critic_target[key] = param_critic[key]
        for key in self.actor_layer_norms:
            param_actor_target[key] = param_actor[key]
        return param_critic_target, param_actor_target

    @partial(jax.jit, static_argnums=0)
    def _explore(self, param_actor, observations, key)-> np.ndarray:
        predictions = self._predict(param_actor, observations)[0]
        noise = self.explore_noise * (jax.random.normal(key, shape=predictions.shape))
        noise = jnp.clip(noise, -self.exploration_noise_clip, self.exploration_noise_clip)
        predictions = predictions + noise
        return predictions.clip(-1., 1.)

    def explore(self, observations, state=None, *args, **kwargs) -> np.ndarray:
        predictions = self._explore(self.param_actor, observations[None], next(self.rng))
        return np.asarray(predictions).astype(self.env.action_space.dtype)

    def predict(self, observations, state=None, *args, **kwargs) -> np.ndarray:
        # policy predict and post process to be numpy
        observations = observations[None]
        actions = np.asarray(self._predict(self.param_actor, observations))[0]
        return actions

    @partial(jax.jit, static_argnums=0)
    def _predict(self, param_actor, observations):
        return self.actor.apply(param_actor, observations)

    @partial(jax.jit, static_argnums=0)
    def actor_loss(self,
                   param_actor: hk.Params,
                   param_critic: hk.Params,
                   obs: jnp.ndarray,
                   taus: jnp.ndarray
                   ):

        actions = self.actor.apply(param_actor, obs)
        taus = self.risk_model(taus, self.risk_param)
        qf = self.critic.apply(param_critic, obs, actions, taus).mean(axis=-1)
        qf = jnp.min(qf, axis=-1)
        return -qf.mean(), None

    @partial(jax.jit, static_argnums=0)
    def sample_taus(self, key):
        return self._sample_taus(key, self.n_quantiles)

    @partial(jax.jit, static_argnums=0)
    def critic_loss(self,
                    param_critic: hk.Params,
                    param_critic_target: hk.Params,
                    param_actor_target: hk.Params,
                    obs: jnp.ndarray,
                    actions: jnp.ndarray,
                    reward: jnp.ndarray,
                    dones: jnp.ndarray,
                    next_obs: jnp.ndarray,
                    key: jax.random.PRNGKey
                    ):
        key1, key2 = jax.random.split(key)
        _, current_taus, _ = self.sample_taus(key1)
        _, next_taus, weight = self.sample_taus(key2)

        target_qf = self.compute_target_qf(param_critic_target,
                                           param_actor_target,
                                           next_obs=next_obs,
                                           next_taus=next_taus,
                                           rewards=reward,
                                           dones=dones,
                                           key=key)
        current_qf = self.critic.apply(param_critic, obs, actions, current_taus)
        loss = jnp.stack([(self.quantile_loss(target_qf, current_qf[:, i, :],  next_taus).sum(axis=-1) * weight).sum(axis=-1)
                         for i in range(self.n_critics)], axis=1).sum(axis=-1).mean()
        return loss, None

    @partial(jax.jit, static_argnums=0)
    def compute_target_qf(self,
                          param_critic_target: hk.Params,
                          param_actor_target: hk.Params,
                          next_obs: jnp.ndarray,
                          next_taus: jnp.ndarray,
                          rewards: jnp.ndarray,
                          dones: jnp.ndarray,
                          key: jax.random.PRNGKey,
                          ):
        key1, key2 = jax.random.split(key, 2)
        next_actions = self.actor.apply(param_actor_target, next_obs)

        noise = self.target_noise * jax.random.normal(key=key2, shape=next_actions.shape)
        noise = jnp.clip(noise, -self.target_noise_clip, self.target_noise_clip)
        next_actions = jnp.clip(noise + next_actions, -1., 1.)
        next_qf = self.critic.apply(param_critic_target, next_obs, next_actions, next_taus)
        next_qf = next_qf.reshape(next_qf.shape[0], -1)
        next_qf = jnp.sort(next_qf, axis=-1)
        if self.drop_per_net > 0:
            next_qf = next_qf[:, :-self.n_critics * self.drop_per_net]
        return jax.lax.stop_gradient(rewards + self.gamma * (1. - dones) * next_qf)

    def train_step(self):
        obs, actions, rewards, dones, next_obs = self.buffer.sample(self.batch_size)

        self.opt_critic_state, self.param_critic, qf_loss, _ = optimize(
            self.critic_loss,
            self.opt_critic,
            self.opt_critic_state,
            self.param_critic,
            self.param_critic_target,
            self.param_actor_target,
            obs, actions, rewards, dones,
            next_obs, next(self.rng))
        self.logger.record(key='train/qf_loss', value=qf_loss.item())

        if self._n_updates % self.delay == 0:
            _, taus_hat, _ = self.sample_taus(key=next(self.rng))
            taus_hat = self.risk_model(taus_hat, self.risk_param)
            self.opt_actor_state, self.param_actor, actor_loss, _ = optimize(
                self.actor_loss,
                self.opt_actor,
                self.opt_actor_state,
                self.param_actor,
                self.param_critic,
                obs, taus_hat)
            self.logger.record(key='train/pi_loss', value=actor_loss.item())
            self.param_critic_target = soft_update(self.param_critic_target, self.param_critic, self.soft_update_coef)
            self.param_actor_target = soft_update(self.param_actor_target, self.param_actor, self.soft_update_coef)

            self.param_critic_target, self.param_actor_target = self.layer_norm_update(
                self.param_critic_target, self.param_critic, self.param_actor_target, self.param_actor
            )

    def save(self, path):
        params = {"param_actor": self.param_actor,
                  "param_critic": self.param_critic
                  }
        return np.savez(path, **params)

    def load(self, path):
        params = np.load(path, allow_pickle=True)
        self.param_actor = params["param_actor"].item()
        self.param_critic = params["param_critic"].item()

    @property
    def named_config(self) -> str:
        return f'IQN_TD3_{self.risk_name}:{self.risk_param}_seed_{self.seed}'
