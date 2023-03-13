import haiku as hk
import jax.numpy as jnp
from typing import Sequence
from risk_sensitive_rl.common_model.commons import MLP
import jax


class CMVCritic(hk.Module):
    def __init__(self,
                 net_arch: Sequence[int] = (256, 256),
                 n_critics: int = 2,
                 ):
        super().__init__()
        self.net_arch = net_arch
        self.n_critics = n_critics

    def __call__(self, observations, actions):
        def qf(observation, action):
            net_results = MLP(
                output_dim=2,
                net_arch=self.net_arch
            )(jnp.concatenate((observation, action), axis=-1))
            return net_results
        results = jnp.stack([qf(observations, actions) for _ in range(self.n_critics)], axis=1)
        qf = results[..., [0]]
        beta_qf = results[..., [1]]

        return qf, jax.nn.relu(beta_qf)


class RewardPredictor(hk.Module):
    def __init__(self,
                 net_arch: Sequence[int] = (256, 256),
                 ):
        super().__init__()
        self.net_arch = net_arch

    def __call__(self, observations, actions):
        r_hat = MLP(
            output_dim=1,
            net_arch=self.net_arch
        )(jnp.concatenate((observations, actions), axis=-1))
        return r_hat

