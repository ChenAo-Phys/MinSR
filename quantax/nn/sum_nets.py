from typing import Optional, Union, Sequence
import numpy as np
import jax
from jaxtyping import Key
import equinox as eqx
from .initializers import he_normal
from .modules import Conv, Sequential
from .activation import SinhShift, Exp, Scale, pair_cpl
from .nqs_layers import ReshapeConv, ConvSymmetrize
from ..symmetry import Symmetry, Identity
from ..global_defs import get_lattice, is_default_cpl, is_params_cpl, get_subkeys


class _ResBlock(eqx.Module):
    """Residual block"""
    conv1: Conv
    conv2: Conv
    nblock: int = eqx.field(static=True)

    def __init__(self, channels: int, kernel_size: int, nblock: int, total_blocks: int):
        def new_layer(is_first_layer: bool, is_last_layer: bool) -> Conv:
            lattice = get_lattice()
            in_channels = lattice.shape[-1] if is_first_layer else channels
            return Conv(
                num_spatial_dims=lattice.dim,
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding="CIRCULAR",
                use_bias=not is_last_layer,
                kernel_init=he_normal,
                key=get_subkeys(),
            )
        
        self.conv1 = new_layer(nblock == 0, False)
        self.conv2 = new_layer(False, nblock == total_blocks - 1)
        self.nblock = nblock

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        residual = x.copy()
        x /= np.sqrt(self.nblock + 1, dtype=x.dtype)

        if self.nblock == 0:
            x /= np.sqrt(2, dtype=x.dtype)
        else:
            x = jax.nn.relu(x)
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        return x + residual


def ResSum(
    depth: int,
    channels: int,
    kernel_size: Union[int, Sequence[int]],
    use_sinh: bool = False,
    trans_symm: Optional[Symmetry] = None,
):
    if is_params_cpl():
        raise ValueError("'ResSum' only supports real parameters.")
    if depth % 2:
        raise ValueError(f"'depth' should be a multiple of 2, got {depth}")
    
    num_blocks = depth // 2
    blocks = [
        _ResBlock(channels, kernel_size, i, num_blocks) for i in range(num_blocks)
    ]

    scale = Scale(1 / np.sqrt(num_blocks))
    layers = [ReshapeConv(), *blocks, scale]

    if is_default_cpl():
        cpl_layer = eqx.nn.Lambda(lambda x: pair_cpl(x))
        layers.append(cpl_layer)
    layers.append(SinhShift() if use_sinh else Exp())
    if trans_symm is not Identity():
        layers.append(ConvSymmetrize(trans_symm))

    return Sequential(layers, holomorphic=False)
