import jax.numpy as jnp
import jax


@jax.jit
def sample_cvar(taus: jnp.ndarray, alpha: jnp.ndarray):
    return alpha * taus


@jax.jit
def sample_cvar_general(taus: jnp.ndarray, alpha: jnp.ndarray):
    return jnp.where(alpha >= 1, (alpha - 1) + (alpha - 1) * taus, alpha * taus)


@jax.jit
def cpw(taus: jnp.ndarray, alpha: jnp.ndarray):
    taus = jnp.clip(taus, 1e-8, 1-1e-8)
    x = jnp.power(taus, alpha)
    y = jnp.power(1. - taus, alpha)
    return x/jnp.power(x + y, 1/alpha)


@jax.jit
def wang(taus: jnp.ndarray, alpha: jnp.ndarray):
    Phi = jax.scipy.stats.norm.cdf
    InvPhi = jax.scipy.stats.norm.ppf
    return Phi(InvPhi(taus) + alpha)


@jax.jit
def sample_power_general(taus: jnp.ndarray, alpha: jnp.ndarray):
    return jnp.where(alpha >= 0, jnp.power(taus, 1/(1 + jnp.abs(alpha))), 1 - jnp.power(1 - taus, 1/(1 + jnp.abs(alpha))))


@jax.jit
def sample_power(taus: jnp.ndarray, alpha: float):
    return 1. - jnp.power((1. - taus), 1 / (1 + jnp.abs(alpha)))

