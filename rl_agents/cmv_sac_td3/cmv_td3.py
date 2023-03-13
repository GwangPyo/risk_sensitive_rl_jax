from rl_agents.td3 import TD3
import gym
from rl_agents.cmv_sac_td3.policy import CMVCritic, RewardPredictor
from typing import Optional, Callable
from utils.optimize import optimize, soft_update

import haiku as hk
import jax.numpy as jnp
import jax
import optax
from functools import partial


class CMVTD3(TD3):
    def __init__(self,
                 env: gym.Env,
                 buffer_size: int = 1000_000,
                 gamma: float = 0.99,
                 batch_size: int = 256,
                 warmup_steps: int = 2000,
                 seed: int = 0,
                 delay: int = 2,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_reward: float = 3e-4,
                 soft_update_coef: float = 5e-2,
                 target_noise=0.3,
                 target_noise_clip=0.5,
                 exploration_noise=0.3,
                 exploration_noise_clip=0.5,
                 risk_param=0.5,
                 actor_fn: Callable = None,
                 critic_fn: Callable = None,
                 wandb: bool = False,
                 n_critis: int = 2,

                 ):
        self.rng = hk.PRNGSequence(seed)
        self.n_critics = n_critis
        self.env = env
        if critic_fn is None:
            def critic_fn(observation, action):
                return CMVCritic(n_critics=n_critis)(observation, action)

            obs_placeholder, a_placeholder = self.make_placeholder()
            self.critic = hk.without_apply_rng(hk.transform(critic_fn))
            self.param_critic = self.param_critic_target = self.critic.init(
                next(self.rng), obs_placeholder, a_placeholder
            )

            opt_init, self.opt_critic = optax.chain(
                optax.add_decayed_weights(1e-2),
                optax.clip_by_global_norm(0.5),
                optax.adabelief(lr_critic))

        def reward_predictor(observation, action):
            return RewardPredictor()(observation, action)

        obs_placeholder, a_placeholder = self.make_placeholder()
        self.reward_predictor = hk.without_apply_rng(hk.transform(reward_predictor))
        self.param_reward_predictor = self.reward_predictor.init(
            next(self.rng), obs_placeholder, a_placeholder
        )

        opt_init, self.opt_reward_predictor = optax.chain(
            optax.add_decayed_weights(1e-2),
            optax.clip_by_global_norm(0.5),
            optax.adabelief(lr_reward))

        self.opt_reward_predictor_state = opt_init(self.param_reward_predictor)

        super().__init__(env,
                         buffer_size,
                         gamma,
                         batch_size,
                         warmup_steps,
                         seed,
                         lr_actor,
                         lr_critic,
                         delay=delay,
                         soft_update_coef=soft_update_coef,
                         target_noise=target_noise,
                         target_noise_clip=target_noise_clip,
                         drop_per_net=1,
                         risk_param=0.5,
                         wandb=wandb,
                         actor_fn=None,
                         critic_fn=critic_fn,
                         explore_noise=exploration_noise,
                         exploration_noise_clip=exploration_noise_clip,
                         n_critics=n_critis)

        self.cmv_beta = risk_param
        self.gamma_square = self.gamma ** 2

    @partial(jax.jit, static_argnums=(0,))
    def reward_predictor_loss(self,
                              param_reward_predictor: hk.Params,
                              observations: jnp.ndarray,
                              actions: jnp.ndarray,
                              rewards: jnp.ndarray):
        rf_hat = self.reward_predictor.apply(param_reward_predictor, observations, actions)
        errors = (rewards - rf_hat) ** 2
        return errors.mean(), jax.lax.stop_gradient(errors)

    @partial(jax.jit, static_argnums=(0,))
    def critic_loss(self,
                    param_critic: hk.Params,
                    param_critic_target: hk.Params,
                    param_actor: hk.Params,
                    obs: jnp.ndarray,
                    actions: jnp.ndarray,
                    reward: jnp.ndarray,
                    dones: jnp.ndarray,
                    next_obs: jnp.ndarray,
                    rewards_error: jnp.ndarray,
                    key: jax.random.PRNGKey
                    ):
        qf, qf_beta = self.critic.apply(param_critic, obs, actions)
        target_qf, target_qf_beta = self.compute_target_qf(
            param_critic_target,
            param_actor,
            next_obs,
            reward, rewards_error,
            dones, key
        )
        qf_loss = ((qf - target_qf[..., None]) ** 2).sum(axis=-2).mean()
        qf_beta_loss = ((qf_beta - target_qf_beta[..., None]) ** 2).sum(axis=-2).mean()
        return qf_loss + qf_beta_loss, (qf_loss, qf_beta_loss)

    @partial(jax.jit, static_argnums=(0,))
    def compute_target_qf(self,
                          param_critic_target: hk.Params,
                          param_actor: hk.Params,
                          next_obs: jnp.ndarray,
                          rewards: jnp.ndarray,
                          reward_error: jnp.ndarray,
                          dones: jnp.ndarray,
                          key: jax.random.PRNGKey,
                          ):
        next_actions = self.actor.apply(param_actor, next_obs)
        noise = jax.random.normal(key, shape=next_actions.shape)
        noise = self.target_noise * noise
        noise = noise.clip(-self.target_noise_clip, self.target_noise_clip)
        next_actions = next_actions + noise
        next_actions = next_actions.clip(-1., 1.)

        next_qf, next_qf_beta = self.critic.apply(param_critic_target, next_obs, next_actions)
        next_qf = next_qf.min(axis=-2)
        next_qf_beta = next_qf_beta.max(axis=-2)

        next_target_qf = jax.lax.stop_gradient(rewards + (1. - dones) * self.gamma * next_qf)
        next_target_qf_beta = jax.lax.stop_gradient(rewards + (1. - dones) * self.gamma_square * next_qf_beta)
        return next_target_qf, next_target_qf_beta

    @partial(jax.jit, static_argnums=(0,))
    def actor_loss(self,
                   param_actor: hk.Params,
                   param_critic: hk.Params,
                   obs: jnp.ndarray,
                   key: jax.random.PRNGKey
                   ):
        actions = self.actor.apply(param_actor, obs)
        qf, qf_beta = self.critic.apply(param_critic, obs, actions)
        qf = qf.min(axis=-2)
        qf_beta = qf.max(axis=-2)

        actor_loss = (-qf + self.cmv_beta * qf_beta).mean()
        return actor_loss, None

    def train_step(self):
        obs, actions, rewards, dones, next_obs = self.buffer.sample(self.batch_size)

        self.opt_reward_predictor_state, self.param_reward_predictor, rf_loss, reward_error = optimize(
            self.reward_predictor_loss,
            self.opt_reward_predictor,
            self.opt_reward_predictor_state,
            self.param_reward_predictor,
            obs, actions, rewards
        )

        self.opt_critic_state, self.param_critic, _, (qf_loss, qf_beta_loss) = optimize(
            self.critic_loss,
            self.opt_critic,
            self.opt_critic_state,
            self.param_critic,
            self.param_critic_target,
            self.param_actor,
            obs, actions, rewards, dones,
            next_obs, reward_error, next(self.rng))

        self.opt_actor_state, self.param_actor, actor_loss, log_pi = optimize(
            self.actor_loss,
            self.opt_actor,
            self.opt_actor_state,
            self.param_actor,
            self.param_critic,
            obs, next(self.rng))

        self.logger.record(key='train/pi_loss', value=actor_loss.item())
        self.logger.record(key='train/qf_loss', value=qf_loss.mean().item())
        self.logger.record(key='train/rf_loss', value=rf_loss.item())
        self.logger.record(key='train/qf_beta_loss', value=qf_beta_loss.mean().item())

        self.param_critic_target = soft_update(self.param_critic_target, self.param_critic, self.soft_update_coef)
