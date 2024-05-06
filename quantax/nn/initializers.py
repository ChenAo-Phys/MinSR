from typing import Callable, Sequence
from functools import partial
import jax
import jax.numpy as jnp
from jax.nn import initializers
from jaxtyping import Key
from ..global_defs import get_params_dtype


variance_scaling = partial(
    initializers.variance_scaling, in_axis=1, out_axis=0, batch_axis=()
)


def _fix_init_axis(initializer: Callable) -> Callable:
    return initializer(in_axis=1, out_axis=0, batch_axis=(), dtype=get_params_dtype())


lecun_normal = _fix_init_axis(initializers.lecun_normal)
lecun_uniform = _fix_init_axis(initializers.lecun_uniform)
glorot_normal = _fix_init_axis(initializers.glorot_normal)
glorot_uniform = _fix_init_axis(initializers.glorot_uniform)
he_normal = _fix_init_axis(initializers.he_normal)
he_uniform = _fix_init_axis(initializers.he_uniform)


def value_pad(value: jax.Array) -> Callable:
    def init(
        key: Key, shape: Sequence, dtype: jnp.dtype = jnp.float_
    ) -> jax.Array:
        if len(value.shape) != len(shape):
            raise ValueError("Only the value with the same dimension can be extended.")

        pad_width = []
        for l_kernel, l_value in zip(shape, value.shape):
            pad_left = (l_kernel - 1) // 2 - (l_value - 1) // 2
            pad_right = l_kernel // 2 - l_value // 2
            pad_width.append((pad_left, pad_right))

        # pad_width = [
        #     (0, l_kernel - l_value) for l_kernel, l_value in zip(shape, value.shape)
        # ]
        kernel = jnp.pad(value, pad_width).astype(dtype)
        return kernel

    return init
