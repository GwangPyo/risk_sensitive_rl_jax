import haiku as hk
import jax.numpy as jnp
import jax.nn as nn
from typing import Sequence
from risk_sensitive_rl.common_model import MLP, IQNHead


class CosineQf(hk.Module):
    def __init__(self,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_cos: int = 256
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
                 n_cos: int = 256,
                 n_critics: int = 2,
                 ):
        super().__init__()
        self.kwargs = {
            "z_dim": z_dim,
            "net_arch": net_arch,
            "n_cos": n_cos
        }
        self.n_critics = n_critics

    def __call__(self, obs, actions, taus):
        args = (obs, actions, taus)
        return jnp.stack([CosineQf(**self.kwargs)(*args) for _ in range(self.n_critics)], axis=1)


class DeterministicActor(hk.Module):
    def __init__(self,
                 actions_dim: int,
                 net_arch: Sequence[int] = (256, 256)
                 ):
        super().__init__()
        self.layers = MLP(output_dim=actions_dim,
                          net_arch=net_arch,
                          activation_fn=nn.relu)
        self.action_dim = actions_dim

    def __call__(self, obs):
        mu = self.layers(obs)
        return nn.tanh(mu)
