import jax.numpy as jnp
import haiku as hk
from typing import Sequence
from risk_sensitive_rl.common_model.commons import MLP, DiscreteActionIQNHead


class IQNPolicy(hk.Module):
    def __init__(self,
                 actions_dim,
                 features_dim,
                 ):
        super().__init__()
        self.actions_dim = actions_dim
        self.features_dim = features_dim

    def __call__(self, observation, taus):
        feature = MLP(output_dim=self.features_dim,
                      net_arch=(256, 256)
                      )(observation)
        head = DiscreteActionIQNHead(self.actions_dim, self.features_dim)
        return head(feature, taus)




