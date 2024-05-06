from typing import Optional, Callable
import jax
import jax.numpy as jnp
from equinox.nn import Lambda
from .modules import Conv, Sequential
from .nqs_layers import ReshapeConv
from ..sites import TriangularB
from ..utils import neel, stripe
from ..global_defs import get_lattice, get_subkeys


def sign(neg: bool = False) -> Callable:
    if neg:
        fn = lambda x: -jnp.sign(jnp.sum(jnp.cos(x)))
    else:
        fn = lambda x: jnp.sign(jnp.sum(jnp.cos(x)))
    return Lambda(fn)


def phase(neg: bool = False) -> Callable:
    def fn(x):
        x = jnp.sum(jnp.exp(1j * x))
        x /= jnp.abs(x)
        if neg:
            x *= -1
        return x

    return Lambda(fn)


def cos(neg: bool = False) -> Callable:
    if neg:
        fn = lambda x: -jnp.mean(jnp.cos(x))
    else:
        fn = lambda x: jnp.mean(jnp.cos(x))
    return Lambda(fn)


class SgnNet(Sequential):
    def __init__(
        self,
        kernel: Optional[jax.Array] = None,
        output: str = "sign",
        neg: bool = False,
    ):
        if kernel is not None:
            def kernel_init(key, shape, dtype):
                return kernel.reshape(shape).astype(dtype)
        elif output == "sign":
            kernel_init = jax.nn.initializers.zeros
        else:
            kernel_init = jax.nn.initializers.normal(jnp.pi / 4)

        if output == "sign":
            actfn = sign(neg)
        elif output == "phase":
            actfn = phase(neg)
        elif output == "cos":
            actfn = cos(neg)
        else:
            raise ValueError

        lattice = get_lattice()
        conv = Conv(
            num_spatial_dims=lattice.dim,
            in_channels=lattice.shape[-1],
            out_channels=1,
            kernel_size=lattice.shape[:-1],
            padding="CIRCULAR",
            use_bias=False,
            kernel_init=kernel_init,
            key=get_subkeys(),
        )
        layers = [ReshapeConv(), conv, actfn]
        super().__init__(layers, holomorphic=False)

    def plot(self):
        import matplotlib.pyplot as plt

        params = jax.tree_util.tree_flatten(self.layers[1])[0][0]
        params = params[0, 0]
        params = params - jnp.round(params / jnp.pi) * jnp.pi
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        pic = ax.imshow(
            params,
            cmap=plt.get_cmap("twilight"),
            interpolation="nearest",
            vmin=-jnp.pi / 2,
            vmax=jnp.pi / 2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(pic, fraction=0.046, pad=0.04)
        return fig


def MarshallSign(output: str = "sign") -> Sequential:
    L = get_lattice().nsites
    neg = (L // 4) % 2 == 1
    return SgnNet(jnp.pi / 4 * neel(), output, neg)


def StripeSign(output: str = "sign", alternate_dim: int = 1) -> Sequential:
    L = get_lattice().nsites
    neg = (L // 4) % 2 == 1
    return SgnNet(jnp.pi / 4 * stripe(alternate_dim), output, neg)


def Neel120(output: str = "phase") -> Sequential:
    lattice = get_lattice()
    Lx, Ly = lattice.shape[:2]
    x = 2 * jnp.arange(Lx)
    if isinstance(lattice, TriangularB):
        y = jnp.zeros(Ly, dtype=x.dtype)
    else:
        y = jnp.arange(Ly)
    kernel = (x[:, None] + y[None, :]) % 3
    kernel = jnp.pi / 3 * kernel - jnp.pi / 6
    return SgnNet(kernel, output)
