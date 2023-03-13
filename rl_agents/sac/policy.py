import haiku as hk
from typing import Sequence
from common_model import MLP
from common_model.commons import IQNHead
import jax.numpy as jnp
from jax import nn


class CosineQf(hk.Module):
    def __init__(self,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_cos: int = 64
                 ):
        super().__init__()

        net_arch = list(net_arch)
        self.feature_embedding = MLP(output_dim=z_dim,
                                     net_arch=net_arch)
        self.iqn_head = IQNHead(
            z_dim=z_dim,
            net_arch=net_arch,
            n_cos=n_cos
        )

    def __call__(self, obs, actions, taus):
        feature = self.feature_embedding(jnp.concatenate((obs, actions), axis=-1))
        qfs = self.iqn_head(feature=feature, taus=taus)
        return qfs


class Critic(hk.Module):
    def __init__(self,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_cos: int = 64
                 ):
        super().__init__()
        self.kwargs = {
            "z_dim": z_dim,
            "net_arch": net_arch,
            "n_cos": n_cos
        }

    def __call__(self, obs, actions, taus):
        args = (obs, actions, taus)
        return jnp.stack([CosineQf(**self.kwargs)(*args) for _ in range(2)], axis=1)


class StochasticActor(hk.Module):
    def __init__(self,
                 actions_dim: int,
                 net_arch: Sequence[int] = (256, 256)
                 ):
        super().__init__()
        net_arch = list(net_arch)
        self.layers = MLP(output_dim=net_arch[-1],
                          net_arch=net_arch[:-1],
                          activation_fn=nn.relu,
                          )
        self.action_dim = actions_dim

    def __call__(self, obs):
        pre = self.layers(obs)
        mu = hk.Linear(self.action_dim)(pre)
        log_sigma = hk.Linear(self.action_dim)(pre)
        log_sigma = jnp.clip(log_sigma, -20, 3)
        return mu, log_sigma
