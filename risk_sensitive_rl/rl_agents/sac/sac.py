from functools import partial
import gym

from risk_sensitive_rl.rl_agents.offpolicy import OffPolicyPG
from risk_sensitive_rl.rl_agents.sac.policy import StochasticActor, Critic
from risk_sensitive_rl.common_model import tanh_normal_reparamterization, get_actions_logprob
from risk_sensitive_rl.utils.optimize import optimize, build_optimizer, soft_update

import numpy as np

import haiku as hk
from typing import Optional, Callable
from risk_sensitive_rl.rl_agents.risk_models import *


class SAC(OffPolicyPG):
    name = "SAC"
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
                 lr_ent: float = 3e-4,
                 ent_coef: Optional[None] = None,
                 soft_update_coef: float = 5e-3,
                 target_entropy: Optional[float] = None,
                 drop_per_net: int = 5,
                 risk_type='cvar',
                 risk_param=1.0,
                 actor_fn: Callable = None,
                 critic_fn: Callable = None,
                 wandb_proj: Optional[str] = None,
                 cfg: Optional[dict] = None,
                 work_dir: Optional[str] = None,
                 n_critics: int = 2,
                 ):
        try:
            self.risk_model = SAC.risk_types[risk_type]
            self.risk_param = risk_param

        except KeyError:
            raise NotImplementedError

        self.risk_name = risk_type
        self.n_critics = n_critics
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
        self.drop_per_net = drop_per_net

        if actor_fn is None:
            def actor_fn(obs):
                return StochasticActor(self.env.action_space.shape[0])(obs)
            obs_placeholder, a_placeholder = self.make_placeholder()
            self.actor = hk.without_apply_rng(hk.transform(actor_fn))
            self.param_actor = self.actor.init(next(self.rng), obs_placeholder)

        if critic_fn is None:
            obs_placeholder, a_placeholder = self.make_placeholder()
            quantile_placeholder = jnp.ones((1, self.n_quantiles))

            def critic_fn(obs, actions, taus):
                return Critic(n_critics=self.n_critics)(obs, actions, taus)

            self.critic = hk.without_apply_rng(hk.transform(critic_fn))
            self.param_critic = self.param_critic_target = self.critic.init(next(self.rng),
                                                                            obs_placeholder,
                                                                            a_placeholder,
                                                                            quantile_placeholder)

        self.log_ent_coef = jnp.asarray([0. ])
        if ent_coef is None:
            if target_entropy is None:
                self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
            else:
                self.target_entropy = target_entropy

            opt_init, self.opt_ent = build_optimizer(lr_ent)
            self.opt_ent_state = opt_init(self.log_ent_coef)
            self.train_ent = True
        else:
            self.log_ent_coef = jnp.asarray([jnp.log(ent_coef + 1e-8)])
            self.train_ent = False

        opt_init, self.opt_actor = build_optimizer(lr_actor)

        self.opt_actor_state = opt_init(self.param_actor)

        opt_init, self.opt_critic = build_optimizer(lr_critic)

        self.opt_critic_state = opt_init(self.param_critic)

        self._n_updates = 0
        self.soft_update_coef = soft_update_coef

        self.critic_layer_norms = []
        for keys in self.param_critic.keys():
            if 'layer_norm' in keys.split('/')[-1]:
                self.critic_layer_norms.append(keys)

    @partial(jax.jit, static_argnums=0)
    def layer_norm_update(self, param_critic_target, param_critic):
        # hard update for the layer norm
        for keys in self.critic_layer_norms:
            param_critic_target[keys] = param_critic[keys]
        return param_critic_target

    def predict(self, observations, state=None, *args, **kwargs) -> np.ndarray:
        # policy predict and post process to be numpy
        observations = observations[None]
        return np.asarray(self._predict(self.param_actor, observations, next(self.rng)))[0]

    @partial(jax.jit, static_argnums=0)
    def _predict(self, param_actor, observations, key):
        return tanh_normal_reparamterization(*self.actor.apply(param_actor, observations), key)

    @partial(jax.jit, static_argnums=0)
    def actor_loss(self,
                   param_actor: hk.Params,
                   param_critic: hk.Params,
                   obs: jnp.ndarray,
                   taus: jnp.ndarray,
                   ent_coef: jnp.ndarray,
                   key: jax.random.PRNGKey
                   ):

        mu, logstd = self.actor.apply(param_actor, obs)
        actions, log_pi = get_actions_logprob(mu, logstd, key)
        taus = self.risk_model(taus, self.risk_param)
        qf = self.critic.apply(param_critic, obs, actions, taus).mean(axis=-1)
        qf = jnp.min(qf, axis=-1, keepdims=True)
        return (ent_coef * log_pi - qf).mean(), log_pi

    @partial(jax.jit, static_argnums=0)
    def critic_loss(self,
                    param_critic: hk.Params,
                    param_critic_target: hk.Params,
                    param_actor: hk.Params,
                    obs: jnp.ndarray,
                    actions: jnp.ndarray,
                    reward: jnp.ndarray,
                    dones: jnp.ndarray,
                    next_obs: jnp.ndarray,
                    ent_coef: jnp.ndarray,
                    key: jax.random.PRNGKey
                    ):
        key1, key2 = jax.random.split(key, 2)
        _, tau_hat, _ = self.sample_taus(key1)
        _, next_tau, weight = self.sample_taus(key2)

        target_qf = self.compute_target_qf(param_critic_target,
                                           param_actor,
                                           next_obs=next_obs,
                                           next_taus=next_tau,
                                           rewards=reward,
                                           dones=dones,
                                           ent_coef=ent_coef,
                                           key=key)

        current_qf = self.critic.apply(param_critic, obs, actions, tau_hat)
        loss = jnp.stack([((self.quantile_loss(target_qf, current_qf[:, i, :],  tau_hat).sum(axis=-1))
                          * weight).sum(axis=-1)
                         for i in range(self.n_critics)], axis=1).sum(axis=-1)
        return loss.mean(), None

    @partial(jax.jit, static_argnums=0)
    def compute_target_qf(self,
                          param_critic_target: hk.Params,
                          param_actor: hk.Params,
                          next_obs: jnp.ndarray,
                          next_taus: jnp.ndarray,
                          rewards: jnp.ndarray,
                          dones: jnp.ndarray,
                          ent_coef: jnp.ndarray,
                          key: jax.random.PRNGKey,
                          ):
        mu, logstd = self.actor.apply(param_actor, next_obs)
        next_actions, next_log_pi = get_actions_logprob(mu, logstd, key)
        next_qf = self.critic.apply(param_critic_target, next_obs, next_actions, next_taus)
        next_qf = next_qf.reshape(next_qf.shape[0], -1)
        next_qf = next_qf.sort(axis=-1)
        if self.drop_per_net > 0:
            next_qf = next_qf[..., :-self.n_critics * self.drop_per_net]
        next_qf = next_qf - ent_coef * next_log_pi
        return jax.lax.stop_gradient(rewards + self.gamma * (1. - dones) * next_qf)

    @partial(jax.jit, static_argnums=0)
    def ent_coef_loss(self,
                      log_ent_coef,
                      current_log_pi
                      ):
        return (-log_ent_coef * jax.lax.stop_gradient(current_log_pi + self.target_entropy).mean()).mean(), None

    def train_step(self):
        obs, actions, rewards, dones, next_obs = self.buffer.sample(self.batch_size)
        ent_coef = jnp.exp(self.log_ent_coef)

        self.opt_critic_state, self.param_critic, qf_loss, _ = optimize(
            self.critic_loss,
            self.opt_critic,
            self.opt_critic_state,
            self.param_critic,
            self.param_critic_target,
            self.param_actor,
            obs, actions, rewards, dones,
            next_obs, ent_coef, next(self.rng))

        _, tau_hat, _ = self.sample_taus(next(self.rng))
        self.opt_actor_state, self.param_actor, actor_loss, log_pi = optimize(
            self.actor_loss,
            self.opt_actor,
            self.opt_actor_state,
            self.param_actor,
            self.param_critic,
            obs, tau_hat, ent_coef, next(self.rng))

        if self.train_ent:
            self.opt_ent_state, self.log_ent_coef, ent_coef_loss, _ = optimize(
                self.ent_coef_loss,
                self.opt_ent,
                self.opt_ent_state,
                self.log_ent_coef,
                log_pi
            )
            self.logger.record(key='train/ent_coef_loss', value=ent_coef_loss.item())

        self.logger.record(key='train/pi_loss', value=actor_loss.item())
        self.logger.record(key='train/qf_loss', value=qf_loss.item())
        self.logger.record(key='etc/current_ent_coef', value=ent_coef.item())

        self.param_critic_target = soft_update(self.param_critic_target, self.param_critic, self.soft_update_coef)
        self.param_critic_target = self.layer_norm_update(self.param_critic_target, self.param_critic)

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
        return f'IQN_SAC_{self.risk_name}:{self.risk_param}_seed_{self.seed}'
