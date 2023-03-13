import haiku as hk
import jax.numpy as jnp
import jax.nn as nn
from typing import Sequence
from risk_sensitive_rl.common_model import MLP


class RawCosines(hk.Module):
    def __init__(self, features_dim: int, n_cos: int = 128):
        super().__init__()
        self.features_dim = features_dim
        self.n_cos = n_cos

        self.cosines = jnp.arange(1, self.n_cos + 1, dtype=jnp.float32) * jnp.pi
        self.cosines = self.cosines.reshape(1, 1, -1)

    def __call__(self, taus):
        taus = jnp.expand_dims(taus, axis=-1)
        cosines = jnp.cos(taus * self.cosines)
        return cosines


class RCDSACQf(hk.Module):
    def __init__(self,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_cos: int = 64
                 ):
        super().__init__()
        net_arch = list(net_arch)
        self.z_dim = z_dim
        self.net_arch = net_arch
        self.n_cos = n_cos
        self.pi = jnp.pi * jnp.arange(1, n_cos + 1, dtype=jnp.float32).reshape(1, 1, self.n_cos)

    def __call__(self, risk_param, obs, actions, cum_p):
        num_quantiles = cum_p.shape[-1]
        def feature_extractor(x):
            return MLP(
                output_dim=self.z_dim,
                net_arch=self.net_arch,
            )(x)

        def cosines(cum_p):
            # Calculate features.
            cosine = jnp.cos(jnp.expand_dims(cum_p, 2) * self.pi).reshape(cum_p.shape[0], -1, self.n_cos)
            return cosine

        def cosine_embeddings(cosines_cump, cosines_beta):
            cosines_beta = jnp.repeat(cosines_beta, axis=1, repeats=cosines_cump.shape[1])
            cats = jnp.concatenate((cosines_cump, cosines_beta), axis=-1)
            return MLP(self.z_dim, net_arch=self.net_arch)(cats).reshape(-1, num_quantiles, self.z_dim)

        feature = feature_extractor(jnp.concatenate((obs, actions), axis=-1))
        z = feature.reshape(-1, 1, self.z_dim)
        cosines_cump = cosines(cum_p)
        cosines_beta = cosines(risk_param)
        phi = cosine_embeddings(cosines_cump, cosines_beta)
        final_features = z * phi
        quantiles = (MLP(1, net_arch=self.net_arch)(final_features)).squeeze(-1)
        return quantiles


class RCDSACCritic(hk.Module):
    def __init__(self,
                 n_critics: int = 2,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_cos: int = 64
                 ):
        super().__init__()
        self.n_critics = n_critics
        self.kwargs = {
            "z_dim": z_dim,
            "net_arch": net_arch,
            "n_cos": n_cos
        }

    def __call__(self, risk_params, obs, actions, taus):
        return jnp.stack([RCDSACQf(**self.kwargs)(risk_params, obs, actions, taus) for _ in range(self.n_critics)], axis=1)


class RCDSACActor(hk.Module):
    def __init__(self,
                 actions_dim,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_cos: int = 64
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.actions_dim = actions_dim
        self.feature_embedding = MLP(output_dim=z_dim,
                                     net_arch=net_arch)
        self.n_cosine = n_cos
        self.net_arch = net_arch
        self.pi = jnp.pi * jnp.arange(1, n_cos + 1, dtype=jnp.float32).reshape(1, 1, n_cos)

    def __call__(self, betas: jnp.ndarray, obs: jnp.ndarray):
        feature = MLP(
            self.z_dim,
            self.net_arch,
        )(obs)

        # Calculate features.
        cosine = jnp.cos(jnp.expand_dims(betas, 2) * self.pi).reshape(-1, self.n_cosine)
        cosine_feature = (MLP(output_dim=self.z_dim, net_arch=self.net_arch)(cosine)).reshape(-1, self.z_dim)

        feature = feature * cosine_feature
        outputs = MLP(self.z_dim, self.net_arch, activation_fn=nn.relu)(feature)

        mean = hk.Linear(self.actions_dim)(outputs)
        log_std = hk.Linear(self.actions_dim)(outputs)
        log_std = jnp.clip(log_std, -20, 3)
        return mean, log_std
