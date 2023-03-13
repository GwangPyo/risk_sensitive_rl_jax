import gym
from risk_sensitive_rl import TD3, SAC, RCDSAC, CMVSAC, CMVTD3
import os
import fire
from typing import Optional


class MainObject(object):
    def __init__(self,
                 env_name: str,
                 learning_steps: int = int(1e+4),
                 save_name: Optional[str] = None):
        self.env_name = env_name
        self.learning_steps = learning_steps
        self.save_name = save_name
        self.__env_name = self.env_name.split('-')[0]
        self.dir_name = self.__env_name.lower()

    def save(self, model, save_name: str):
        try:
            os.makedirs(self.dir_name)
        except FileExistsError:
            pass
        model.save("{}/{}".format(self.dir_name, save_name))

    def sac(self,
            buffer_size: int = 1000_000,
            gamma: float = 0.99,
            batch_size: int = 128,
            warmup_steps: int = 10000,
            seed: int = 0,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_ent: float = 3e-4,
            soft_update_coef: float = 5e-2,
            target_entropy: Optional[float] = None,
            drop_per_net: int = 2,
            risk_type='cvar',
            risk_param=1.0,
            wandb: bool = False,
            ):
        env = gym.make(self.env_name)
        model = SAC(env,
                    buffer_size=buffer_size,
                    gamma=gamma,
                    batch_size=batch_size,
                    warmup_steps=warmup_steps,
                    seed=seed,
                    target_entropy=target_entropy,
                    wandb=wandb,
                    lr_critic=lr_critic,
                    lr_actor=lr_actor,
                    lr_ent=lr_ent,
                    risk_type=risk_type,
                    risk_param=risk_param,
                    drop_per_net=drop_per_net,
                    soft_update_coef=soft_update_coef
                    )
        model.learn(self.learning_steps)
        if self.save_name is None:
            save_name = 'iqn_sac_{}_{}_seed_{}'.format(risk_type, risk_param, seed)
        else:
            save_name = self.save_name
        self.save(model, save_name)

    def td3(self,
            buffer_size: int = 1000_000,
            gamma: float = 0.99,
            batch_size: int = 128,
            warmup_steps: int = 10000,
            seed: int = 0,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            delay: int = 2,
            soft_update_coef: float = 5e-2,
            target_noise: float = 0.3,
            target_noise_clip: float = 0.5,
            drop_per_net: int = 0,
            risk_type: str = 'cvar',
            risk_param: float = 1.,
            wandb: bool = False,
            explore_noise: float = 0.3,
            exploration_noise_clip: float = 0.5,
            n_critics: int = 2,
            ):
        env = gym.make(self.env_name)
        model = TD3(env,
                    buffer_size=buffer_size,
                    gamma=gamma,
                    batch_size=batch_size,
                    seed=seed,
                    warmup_steps=warmup_steps,
                    delay=delay,
                    soft_update_coef=soft_update_coef,
                    target_noise=target_noise,
                    target_noise_clip=target_noise_clip,
                    wandb=wandb,
                    explore_noise=explore_noise,
                    exploration_noise_clip=exploration_noise_clip,
                    n_critics=n_critics,
                    lr_critic=lr_critic,
                    lr_actor=lr_actor,
                    risk_type=risk_type,
                    risk_param=risk_param,
                    drop_per_net=drop_per_net,
                    )
        model.learn(self.learning_steps)
        if self.save_name is None:
            save_name = 'iqn_td3_{}_{}_seed_{}'.format(risk_type, risk_param, seed)
        else:
            save_name = self.save_name
        self.save(model, save_name)

    def rcdsac(self,
               buffer_size: int = 1000_000,
               gamma: float = 0.99,
               batch_size: int = 128,
               warmup_steps: int = 10000,
               seed: int = 0,
               lr_actor: float = 3e-4,
               lr_critic: float = 3e-4,
               lr_ent: float = 3e-4,
               soft_update_coef: float = 5e-3,
               target_entropy: Optional[float] = None,
               drop_per_net: int = 2,
               wandb: bool = False,
               risk_type: str = 'cvar',
               min_risk_param: float = 0.,
               max_risk_param: float = 1.,
               ):
        env = gym.make(self.env_name)
        model = RCDSAC(
            env=env,
            buffer_size=buffer_size,
            gamma=gamma,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            seed=seed,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_ent=lr_ent,
            soft_update_coef=soft_update_coef,
            target_entropy=target_entropy,
            drop_per_net=drop_per_net,
            wandb=wandb,
            risk_type=risk_type,
            min_risk_param=min_risk_param,
            max_risk_param=max_risk_param,
        )
        model.learn(self.learning_steps)
        if self.save_name is None:
            save_name = 'rcdsac_{}_seed_{}'.format(risk_type, seed)
        else:
            save_name = self.save_name
        self.save(model, save_name)

    def cmv_sac(self,
                buffer_size: int = 1000_000,
                gamma: float = 0.99,
                batch_size: int = 128,
                warmup_steps: int = 10000,
                seed: int = 0,
                lr_actor: float = 3e-4,
                lr_critic: float = 3e-4,
                lr_ent: float = 3e-4,
                lr_reward: float = 3e-4,
                soft_update_coef: float = 5e-2,
                target_entropy: Optional[float] = None,
                risk_param=0.5,
                wandb: bool = False,
                n_critics: int = 2,
                ):
        env = gym.make(self.env_name)
        model = CMVSAC(
            env=env,
            buffer_size=buffer_size,
            gamma=gamma,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            seed=seed,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_ent=lr_ent,
            lr_reward=lr_reward,
            soft_update_coef=soft_update_coef,
            target_entropy=target_entropy,
            risk_param=risk_param,
            wandb=wandb,
            n_critics=n_critics
        )
        model.learn(self.learning_steps)
        if self.save_name is None:
            save_name = 'cmv_sac_{}_seed_{}'.format(risk_param, seed)
        else:
            save_name = self.save_name
        self.save(model, save_name)

    def cmv_td3(self,
                buffer_size: int = 1000_000,
                gamma: float = 0.99,
                batch_size: int = 128,
                warmup_steps: int = 10000,
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
                wandb: bool = False,
                n_critics: int = 2,
                ):
        env = gym.make(self.env_name)
        model = CMVTD3(env=env,
                       buffer_size=buffer_size,
                       gamma=gamma,
                       batch_size=batch_size,
                       warmup_steps=warmup_steps,
                       seed=seed,
                       delay=delay,
                       lr_actor=lr_actor,
                       lr_critic=lr_critic,
                       lr_reward=lr_reward,
                       soft_update_coef=soft_update_coef,
                       target_noise=target_noise,
                       target_noise_clip=target_noise_clip,
                       exploration_noise_clip=exploration_noise_clip,
                       exploration_noise=exploration_noise,
                       risk_param=risk_param,
                       wandb=wandb,
                       n_critics=n_critics)
        model.learn(self.learning_steps)
        if self.save_name is None:
            save_name = 'cmv_td3_{}_seed_{}'.format(risk_param, seed)
        else:
            save_name = self.save_name
        self.save(model, save_name)


if __name__ == '__main__':
    fire.Fire(MainObject)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
