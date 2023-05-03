from typing import Any, Tuple
import jax
import jax.numpy as jnp
from functools import partial
import optax
import haiku as hk

import optax
from typing import Optional, Callable


def build_optimizer(lr: float,
                    optim_cls: Callable = optax.adabelief,
                    centralize: bool = False,
                    weight_decay: Optional[float] = None,
                    clip_grad_norm: Optional[float] = None,
                    decay_steps: Optional[int] = 500_000
                    ) -> optax.chain:
    args = []
    if centralize is not None:
        args.append(optax.centralize())
    if weight_decay is not None:
        args.append(optax.add_decayed_weights(weight_decay))
    if clip_grad_norm is not None:
        args.append(optax.clip_by_global_norm(clip_grad_norm))
    if decay_steps is not None and decay_steps > 0:
        lr = optax.cosine_decay_schedule(lr, decay_steps=decay_steps)
    args.append(optim_cls(lr))
    return optax.chain(*args)


@partial(jax.jit, static_argnums=(0, 1))
def optimize(
    fn_loss: Any,
    opt: Any,
    opt_state: Any,
    params_to_update: Any,
    *args,
    **kwargs,
) -> Tuple[Any, Any, jnp.ndarray, Any]:
    (loss, aux), grad = jax.value_and_grad(fn_loss, has_aux=True)(
        params_to_update,
        *args,
        **kwargs,
    )
    update, opt_state = opt(grad, opt_state, params=params_to_update)
    params_to_update = optax.apply_updates(params_to_update, update)
    return opt_state, params_to_update, loss, aux


@jax.jit
def soft_update(
    target_params: hk.Params,
    online_params: hk.Params,
    tau: float,
) -> hk.Params:
    """
    Update target network using Polyak-Ruppert Averaging.
    """
    return optax.incremental_update(new_tensors=online_params, old_tensors=target_params,
                                    step_size=tau)
