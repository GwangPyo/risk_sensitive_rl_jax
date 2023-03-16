import optax
from typing import Optional


def build_optimizer(lr: float,
                    optim_cls: optax.adam,
                    centralize:bool = True,
                    weight_decay: Optional[float] = None,
                    clip_grad_norm: Optional[float] = None) -> optax.chain:
    args = []
    if centralize is not None:
        args.append(optax.centralize())
    if weight_decay is not None:
        args.append(optax.add_decayed_weights(weight_decay))
    if clip_grad_norm is not None:
        args.append(optax.clip_by_global_norm(clip_grad_norm))
    args.append(optim_cls(lr))
    return optax.chain(*args)

