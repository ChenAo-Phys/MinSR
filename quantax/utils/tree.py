import jax
from jaxtyping import PyTree
import jax.tree_util as jtu
import jax.flatten_util as jfu
import equinox as eqx
import jax.tree_util as jtu
from jax.lax import with_sharding_constraint
from jax.sharding import PositionalSharding


def tree_fully_flatten(tree: PyTree) -> jax.Array:
    array, unravel_fn = jfu.ravel_pytree(tree)
    return array


def filter_shard(tree: PyTree) -> PyTree:
    sharding = PositionalSharding(jax.local_devices())
    count = sharding.shape[0]
    vals, tree_def = jtu.tree_flatten(tree)
    new_vals = []
    for v in vals:
        if eqx.is_array(v):
            shape = (count,) + (1,) * (v.ndim - 1)
            new_vals.append(with_sharding_constraint(v, sharding.reshape(shape)))
        else:
            new_vals.append(v)
    return jtu.tree_unflatten(tree_def, new_vals)


def filter_replicate(tree: PyTree) -> PyTree:
    sharding = PositionalSharding(jax.local_devices()).replicate()
    dynamic, static = eqx.partition(tree, eqx.is_array)
    dynamic = with_sharding_constraint(dynamic, sharding)
    return eqx.combine(dynamic, static)
