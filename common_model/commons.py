import haiku as hk
import jax.numpy as jnp
import jax.nn as nn
import jax
import numpy as np
from typing import Sequence


@jax.jit
def leaky_softplus(x, p):
    return p * x + (1 - p) * nn.softplus(x)


class MLP(hk.Module):
    def __init__(
        self,
        output_dim: int,
        net_arch: Sequence[int],
        activation_fn=nn.relu,
        squashed_output: bool = False,
        d2rl=False,
        layer_norm=True,
        with_bias=True,
    ):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.net_arch = net_arch
        self.hidden_activation = activation_fn
        self.squashed_output = squashed_output
        self.d2rl = d2rl
        self.hidden_kwargs = {"with_bias": with_bias}
        self.output_kwargs = {"with_bias": with_bias}
        self.layer_norm = layer_norm

    def __call__(self, x):
        x_input = x
        for i, unit in enumerate(self.net_arch):
            x = hk.Linear(unit, **self.hidden_kwargs)(x)
            if self.layer_norm:
                x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = self.hidden_activation(x)
            if self.d2rl and i + 1 != len(self.net_arch):
                x = jnp.concatenate([x, x_input], axis=-1)
        x = hk.Linear(self.output_dim, **self.output_kwargs)(x)
        if self.squashed_output:
            x = nn.tanh(x)
        return x


class CosineEmbeddingNetwork(hk.Module):
    def __init__(self, features_dim: int, n_cos: int = 128):
        super().__init__()
        self.features_dim = features_dim
        self.n_cos = n_cos
        self.linear = hk.Linear(features_dim)

        self.cosines = jnp.arange(1, self.n_cos + 1, dtype=jnp.float32) * jnp.pi
        self.cosines = self.cosines.reshape(1, 1, -1)

    def __call__(self, taus):
        taus = jnp.expand_dims(taus, axis=-1)
        cosines = jnp.cos(taus * self.cosines)
        return nn.relu(self.linear(cosines))


class IQNHead(hk.Module):
    def __init__(self,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_cos: int = 256
                 ):
        super().__init__()

        net_arch = list(net_arch)
        self.feature_embedding = MLP(output_dim=z_dim,
                                     net_arch=net_arch)

        self.cosine_embedding = CosineEmbeddingNetwork(
            features_dim=z_dim,
            n_cos=n_cos,
            )

        self.outputs = MLP(
            net_arch=net_arch,
            output_dim=1
        )

    def __call__(self, feature, taus):
        relu_feature = jax.nn.relu(feature)
        taus = self.cosine_embedding(taus)
        qfs = self.outputs(relu_feature[:, None, :] * taus).squeeze(-1)
        return qfs


class PositiveLinear(hk.Linear):
    def __call__(
            self,
            inputs: jnp.ndarray,
            *,
            precision=None,
    ) -> jnp.ndarray:
        """Computes a linear transform of the input."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1. / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)
        out = jnp.dot(inputs, jnp.abs(w), precision=precision)
        return out


class PowerEmbeddingNet(hk.Module):
    def __init__(self, features_dim: int, n_pow: int = 64):
        super(PowerEmbeddingNet, self).__init__()
        self.features_dim = features_dim
        self.n_pow = n_pow
        self.weight_net = PositiveLinear(features_dim)

        p1 = jnp.arange(1, self.n_pow // 2 + 1, dtype=jnp.float32) / self.n_pow
        p2 = jnp.arange(1, self.n_pow - (self.n_pow // 2) + 1, dtype=jnp.float32)
        self.pow = (jnp.concatenate((p1.reshape(1, -1, 1), p2.reshape(1, -1, 1)), axis=1))

    def __call__(self, taus):
        batch_size = taus.shape[0]
        n_tau = taus.shape[1]
        taus = jnp.expand_dims(taus, axis=1)
        powers = jnp.power(taus, self.pow)
        powers = jnp.transpose(powers, (0, 2, 1))
        powers = powers.reshape(batch_size * n_tau, -1)
        t = PositiveLinear(self.features_dim)(powers)
        return t.reshape((batch_size, n_tau, -1))


class MonotoneMLP(MLP):
    def __init__(
        self,
        output_dim: int,
        net_arch: Sequence[int],
        activation_fn=nn.relu,
        squashed_output: bool = False,
        hidden_scale=1.0,
        output_scale=1.0,
        d2rl=False,
    ):
        super().__init__(
            output_dim,
            net_arch,
            activation_fn,
            squashed_output,
            hidden_scale,
            output_scale,
            d2rl,
            layer_norm=False,
            with_bias=False,
        )

    def __call__(self, x):
        x_input = x
        for i, unit in enumerate(self.net_arch):
            x = PositiveLinear(unit, **self.hidden_kwargs)(x)
            x = self.hidden_activation(x)
            if self.d2rl and i + 1 != len(self.net_arch):
                x = jnp.concatenate([x, x_input], axis=-1)
        x = PositiveLinear(self.output_dim, **self.output_kwargs)(x)
        if self.squashed_output:
            x = nn.tanh(x)
        return x


class MonotoneTransformLayer(hk.Module):
    def __init__(self,
                 n_transform=4,
                 net_arch: Sequence[int] = (64, 64)):

        super().__init__()
        self.n_transform = n_transform
        self.net_arch = net_arch

    def __call__(self, features, taus):

        # taus = taus.clip(1e-6, 1 - 1e-6)

        transform_params = MLP(self.n_transform * 5, net_arch=self.net_arch)(features)
        log_scale, bias, log_alpha, log_sbar, b_bar = jnp.split(transform_params, 5, axis=-1)

        # sample n_transform parameters by neural net
        scale = nn.softplus(log_scale)
        alpha = nn.softplus(log_alpha)
        sbar = nn.softplus(log_sbar)

        for i in range(self.n_transform):
            s = scale[..., [i]]
            b = bias[..., [i]]
            a = alpha[..., [i]]
            sprime = sbar[..., [i]]
            bprime = b_bar[..., [i]]
            taus_before = taus
            taus = s * taus + b
            taus = leaky_softplus(taus, a) + (sprime * taus_before + bprime)
        return taus


class PowerIQNHead(hk.Module):
    def __init__(self,
                 z_dim: int = 256,
                 net_arch: Sequence[int] = (256, 256),
                 n_pow: int = 256
                 ):
        super().__init__()

        net_arch = list(net_arch)

        # self.power_embedding = PowerEmbeddingNet(features_dim=z_dim, n_pow=n_pow)

        self.outputs = MonotoneTransformLayer(
            n_transform=4,
            net_arch=net_arch,
        )

        self.monotone = MonotoneMLP(output_dim=1, net_arch=net_arch)
        self.scale_bias = MLP(output_dim=2, net_arch=net_arch)

    def __call__(self, feature, taus):
        relu_feature = jax.nn.relu(feature)
        return self.outputs(feature, taus)


class FourierHead(hk.Module):
    def __init__(self,
                 features_dim: int,
                 fourier_feature_dim: int = 1024,
                 init_sigma=0.1,
                 net_arch: Sequence[int] = (256, 256)
                 ):
        super().__init__()
        self.features_dim = features_dim
        self.fourier_feature_dim = fourier_feature_dim // 2
        self.init_sigma = init_sigma
        self.net_arch = net_arch

    def __call__(self, feature, taus):
        fourier_feature = hk.Linear(output_size=self.fourier_feature_dim,
                                    w_init=hk.initializers.RandomNormal(stddev=self.init_sigma),
                                    with_bias=False
                                    )(jnp.expand_dims(taus, axis=-1))
        fourier_feature = 2 * jnp.pi * fourier_feature
        cosines = jnp.cos(fourier_feature)
        sines = jnp.sin(fourier_feature)
        taus = hk.Linear(self.features_dim, )(jnp.concatenate((sines, cosines), axis=-1))
        return MLP(output_dim=1, net_arch=self.net_arch)(jnp.expand_dims(feature, axis=-2) * taus).squeeze(axis=-1)


class Conv(hk.Module):
    def __init__(self, in_channel=3, latent_size=256, ):
        super().__init__()




@jax.jit
def gaussian_reparameterization(mu, logstd, key):
    std = jnp.exp(logstd)
    noise = jax.random.normal(key=key, shape=std.shape)
    return mu + noise * std, noise


@jax.jit
def tanh_normal_reparamterization(mu, logstd, key):
    return jnp.tanh(gaussian_reparameterization(mu, logstd, key)[0])


@jax.jit
def get_actions_logprob(mu, logstd, key):
    gaussian_actions, noise = gaussian_reparameterization(mu, logstd, key)
    actions = jnp.tanh(gaussian_actions)
    log_prob_normal = -((gaussian_actions - mu) ** 2) / (2 * jnp.exp(logstd)) - logstd - jnp.log(jnp.sqrt(2 * jnp.pi))
    tanh_adjust = jnp.log(jnp.clip(1. - jnp.power(actions, 2), 1e-6, 1.))
    log_prob = jnp.sum(log_prob_normal, axis=-1, keepdims=True) - jnp.sum(tanh_adjust, axis=-1, keepdims=True)
    return actions, log_prob
