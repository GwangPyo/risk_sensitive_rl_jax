import jax.numpy as jnp
import jax


@jax.jit
def averse_cvar(taus, alpha: float):
    return taus * alpha


@jax.jit
def seeking_cvar(taus, alpha: float):
    return alpha + taus * alpha


@jax.jit
def power(taus, alpha: float):
    alpha = jnp.ones_like(taus) * alpha
    eta = jnp.abs(alpha)
    return jnp.where(alpha >= 0,
                     jnp.power(taus, 1/(1 + eta)),
                     1. - jnp.power((1 - taus), 1/(1 + eta)))


@jax.jit
def wang(taus, alpha: float):
    return jax.scipy.stats.norm.cdf(jax.scipy.stats.norm.ppf(taus) + alpha)


@jax.jit
def cpw(taus, alpha):
    eps = 1e-6
    taus = jnp.clip(taus, eps, 1.- eps)
    taus_eta = jnp.power(taus, alpha)
    one_minus_taus = jnp.power(1. - taus, alpha)
    one_minus_taus = jnp.where(taus == 1, jnp.zeros_like(taus), one_minus_taus)
    return taus_eta / jnp.power((taus_eta + one_minus_taus), 1. - alpha)


@jax.jit
def neutral(taus, *args):
    return taus


if __name__ == '__main__':
    import os
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    print(cpw(jnp.linspace(0, 1, 32), 0.71))
    print(cpw(jnp.linspace(0, 1, 32), -0.71))