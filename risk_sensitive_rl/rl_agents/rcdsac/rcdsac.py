from functools import partial
import gym
import jax.lax

from risk_sensitive_rl.rl_agents.sac import SAC
from risk_sensitive_rl.rl_agents.rcdsac.policy import RCDSACCritic, RCDSACActor
from risk_sensitive_rl.common_model import tanh_normal_reparamterization, get_actions_logprob
from risk_sensitive_rl.utils.optimize import optimize, soft_update
from risk_sensitive_rl.rl_agents.risk_models import *

import numpy as np
import haiku as hk

from typing import Optional, Callable

class RCDSAC(SAC):
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
                 soft_update_coef: float = 5e-3,
                 target_entropy: Optional[float] = None,
                 actor_fn: Callable = None,
                 critic_fn: Callable = None,
                 drop_per_net: int = 5,
                 wandb_proj: Optional[str] = None,
                 cfg: Optional[dict] = None,
                 work_dir: Optional[str] = None,
                 risk_type: str = 'cvar',
                 min_risk_param: float = 0.,
                 max_risk_param: float = 1.,
                 n_critics: int = 2,

                 ):
        try:
            self.risk_model = RCDSAC.risk_types[risk_type]
        except KeyError:
            raise NotImplementedError
        self.risk_name = risk_type
        self.rng = hk.PRNGSequence(seed)
        self.env = env
        n_quantiles = 32
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        if actor_fn is None:
            def actor_fn(risk_param, obs):
                return RCDSACActor(self.env.action_space.shape[0])(risk_param, obs)

            obs_placeholder, a_placeholder = self.make_placeholder()
            risk_param_placeholder = jnp.ones((1, 1))
            self.actor = hk.without_apply_rng(hk.transform(actor_fn))
            self.param_actor = self.actor.init(next(self.rng), risk_param_placeholder, obs_placeholder)

        if critic_fn is None:

            def critic_fn(risk_param, obs, actions, taus):
                return RCDSACCritic(n_critics=n_critics)(risk_param, obs, actions, taus)

            obs_placeholder, a_placeholder = self.make_placeholder()
            risk_param_placeholder = jnp.ones((1, 1))
            quantile_placeholder = jnp.ones((1, self.n_quantiles))
            self.critic = hk.without_apply_rng(hk.transform(critic_fn))
            self.param_critic = self.param_critic_target = self.critic.init(next(self.rng),
                                                                            risk_param_placeholder,
                                                                            obs_placeholder,
                                                                            a_placeholder,
                                                                            quantile_placeholder)

        super().__init__(env,
                         buffer_size=buffer_size,
                         gamma=gamma,
                         batch_size=batch_size,
                         warmup_steps=warmup_steps,
                         seed=seed,
                         target_entropy=target_entropy,
                         actor_fn=actor_fn,
                         critic_fn=critic_fn,
                         lr_actor=lr_actor,
                         lr_critic=lr_critic,
                         lr_ent=lr_ent,
                         wandb_proj=wandb_proj,
                         work_dir=work_dir,
                         cfg=cfg)

        self._n_updates = 0
        self.soft_update_coef = soft_update_coef
        self.drop_per_net = drop_per_net

        self.min_risk_param, self.max_risk_param = min_risk_param, max_risk_param
        self.current_risk = self.set_alpha()

    def set_alpha(self):
        return self.sample_alpha(key=next(self.rng), batch_size=1)

    def sample_alpha(self,
                     key: jax.random.PRNGKey,
                     batch_size: int):
        scale = self.max_risk_param - self.min_risk_param
        uniform = jax.random.uniform(key=key, shape=(batch_size, 1))
        alpha = scale * uniform + self.min_risk_param
        return alpha

    def predict(self, observations: jnp.ndarray, state=None, *args, **kwargs) -> np.ndarray:
        # policy predict and post process to be numpy
        observations = observations[None]
        if state is None:
            state = self.current_risk
        else:
            state = jnp.asarray([state], dtype=jnp.float32)[None]
        return np.asarray(self._predict(self.param_actor, risk_param=state,
                                        observations=observations, key=next(self.rng)))[0]

    def explore(self, observations, state=None, *args, **kwargs) -> np.ndarray:
        return self.predict(observations, state, *args, **kwargs)

    @partial(jax.jit, static_argnums=0)
    def _predict(self, param_actor, risk_param, observations, key):
        return tanh_normal_reparamterization(*self.actor.apply(param_actor, risk_param, observations), key)

    @partial(jax.jit, static_argnums=0)
    def actor_loss(self,
                   param_actor: hk.Params,
                   param_critic: hk.Params,
                   obs: jnp.ndarray,
                   taus: jnp.ndarray,
                   ent_coef: jnp.ndarray,
                   alpha: jnp.ndarray,
                   key: jax.random.PRNGKey
                   ):

        mu, logstd = self.actor.apply(param_actor, risk_param=alpha, obs=obs)
        actions, log_pi = get_actions_logprob(mu, logstd, key)
        taus = self.risk_model(taus, alpha)
        qf = self.critic.apply(param_critic,
                               risk_param=alpha, obs=obs,
                               actions=actions, taus=taus).mean(axis=-1)
        qf = jnp.min(qf, axis=-1, keepdims=True)
        return (ent_coef * log_pi - qf).mean(), jax.lax.stop_gradient(log_pi)

    @partial(jax.jit, static_argnums=0)
    def critic_loss(self,
                    param_critic: hk.Params,
                    param_critic_target: hk.Params,
                    param_actor: hk.Params,
                    obs: jnp.ndarray,
                    actions: jnp.ndarray,
                    rewards: jnp.ndarray,
                    dones: jnp.ndarray,
                    next_obs: jnp.ndarray,
                    ent_coef: jnp.ndarray,
                    alpha: jnp.ndarray,
                    key: jax.random.PRNGKey
                    ):

        key1, key2 = jax.random.split(key)
        _, taus_hat, _ = self.sample_taus(key1)
        _, next_taus, weight = self.sample_taus(key2)

        target_qf = self.compute_target_qf(param_critic_target,
                                           param_actor,
                                           next_obs=next_obs,
                                           next_taus=next_taus,
                                           rewards=rewards,
                                           dones=dones,
                                           ent_coef=ent_coef,
                                           alpha=alpha,
                                           key=key)

        current_qf = self.critic.apply(param_critic, risk_param=alpha,
                                       obs=obs, actions=actions, taus=taus_hat)

        loss = jnp.stack([(self.quantile_loss(target_qf,
                                              current_qf[:, i, :],
                                              taus_hat).sum(axis=-1) * weight).sum(axis=-1)
                          for i in range(self.n_critics)], axis=1)
        return loss.sum(axis=-1).mean(), target_qf

    @partial(jax.jit, static_argnums=0)
    def compute_target_qf(self,
                          param_critic_target: hk.Params,
                          param_actor: hk.Params,
                          next_obs: jnp.ndarray,
                          next_taus: jnp.ndarray,
                          rewards: jnp.ndarray,
                          dones: jnp.ndarray,
                          ent_coef: jnp.ndarray,
                          alpha: jnp.ndarray,
                          key: jax.random.PRNGKey,
                          ):
        mu, logstd = self.actor.apply(param_actor, risk_param=alpha, obs=next_obs)
        next_actions, next_log_pi = get_actions_logprob(mu, logstd, key)
        next_qf = self.critic.apply(param_critic_target, risk_param=alpha,
                                    obs=next_obs, actions=next_actions, taus=next_taus)

        next_qf = next_qf.reshape(next_qf.shape[0], -1).sort(axis=-1)

        if self.drop_per_net > 0:
            next_qf = next_qf[..., :-2 * self.drop_per_net]
        next_qf = next_qf - ent_coef * next_log_pi
        return jax.lax.stop_gradient(rewards + self.gamma * (1. - dones) * next_qf)

    def done_callback(self):
        self.current_risk = self.set_alpha()

    @partial(jax.jit, static_argnums=0)
    def ent_coef_loss(self,
                      log_ent_coef,
                      current_log_pi
                      ):
        return (-log_ent_coef * jax.lax.stop_gradient(current_log_pi + self.target_entropy)).mean(), None

    def train_step(self):
        obs, actions, rewards, dones, next_obs = self.buffer.sample(self.batch_size)

        alpha = self.sample_alpha(key=next(self.rng), batch_size=self.batch_size)
        ent_coef = jax.lax.stop_gradient(jnp.exp(self.log_ent_coef))
        self.opt_critic_state, self.param_critic, qf_loss, target_qf = optimize(
            self.critic_loss,
            self.opt_critic,
            self.opt_critic_state,
            params_to_update=self.param_critic,
            param_critic_target=self.param_critic_target,
            param_actor=self.param_actor,
            obs=obs, actions=actions,
            rewards=rewards,
            dones=dones,
            next_obs=next_obs,
            ent_coef=ent_coef,
            alpha=alpha,
            key=next(self.rng))

        _, taus, _ = self.sample_taus(next(self.rng), )
        self.opt_actor_state, self.param_actor, actor_loss, log_pi = optimize(
            self.actor_loss,
            self.opt_actor,
            self.opt_actor_state,
            self.param_actor,
            param_critic=self.param_critic,
            obs=obs, taus=taus, ent_coef=ent_coef, alpha=alpha, key=next(self.rng))

        self.opt_ent_state, self.log_ent_coef, ent_coef_loss, _ = optimize(
            self.ent_coef_loss,
            self.opt_ent,
            self.opt_ent_state,
            self.log_ent_coef,
            log_pi
        )

        self.param_critic_target = soft_update(self.param_critic_target, self.param_critic, self.soft_update_coef)
        self.logger.record(key='train/pi_loss', value=actor_loss.item())
        self.logger.record(key='train/qf_loss', value=qf_loss.item())
        self.logger.record(key='train/ent_coef_loss', value=ent_coef_loss.item())
        self.logger.record(key='etc/current_ent_coef', value=ent_coef.item())

    def make_placeholder(self):
        a = self.env.action_space.sample()
        s = self.env.observation_space.sample()
        return jnp.asarray(s[None]), jnp.asarray(a[None])

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
        return f'RCDSAC_{self.risk_name}_seed_{self.seed}'
