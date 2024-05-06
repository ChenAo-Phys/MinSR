from typing import Optional, Sequence, Tuple, Union, Callable, Any
from jaxtyping import Key
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.nn import initializers
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx
from .initializers import lecun_normal
from ..global_defs import get_params_dtype


class Linear(eqx.Module):
    """Performs a linear transformation."""

    weight: jax.Array
    bias: Optional[jax.Array]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        kernel_init: Callable = lecun_normal,
        bias_init: Callable = initializers.zeros,
        *,
        key: Key,
    ):
        """**Arguments:**
        - `in_features`: The input size.
        - `out_features`: The output size.
        - `use_bias`: Whether to add on a bias as well.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        wkey, bkey = jrandom.split(key, 2)
        dtype = get_params_dtype()
        self.weight = kernel_init(wkey, (out_features, in_features), dtype)
        if use_bias:
            self.bias = bias_init(bkey, (out_features,), dtype)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    @jax.named_scope("eqx.nn.Linear")
    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        """**Arguments:**
        - `x`: The input. Should be a JAX array of shape `(in_features,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        !!! info
            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.
        **Returns:**
        A JAX array of shape `(out_features,)`
        """

        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        return x


def _ntuple(n: int) -> Callable:
    def parse(x: Union[int, Sequence]) -> tuple:
        if isinstance(x, Sequence):
            if len(x) == n:
                return tuple(x)
            else:
                raise ValueError(
                    f"Length of {x} (length = {len(x)}) is not equal to {n}"
                )
        elif isinstance(x, int):
            return (x,) * n
        else:
            raise ValueError(f"{x} should be a Sequence or int, got {type(x)}")

    return parse


class Conv(eqx.Module):
    """General N-dimensional convolution."""

    num_spatial_dims: int = eqx.field(static=True)
    weight: jax.Array
    bias: Optional[jax.Array]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_size: Tuple[int, ...] = eqx.field(static=True)
    stride: Tuple[int, ...] = eqx.field(static=True)
    padding: Union[str, Tuple[Tuple[int, int], ...]] = eqx.field(static=True)
    dilation: Tuple[int, ...] = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = "CIRCULAR",
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        kernel_init: Callable = lecun_normal,
        bias_init: Callable = initializers.zeros,
        *,
        key: Key,
        **kwargs,
    ):
        """**Arguments:**
        - `num_spatial_dims`: The number of spatial dimensions. For example traditional
            convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each spatial
            dimension.
        - `dilation`: The dilation of the convolution.
        - `groups`: The number of input channel groups. At `groups=1`,
            all input channels contribute to all output channels. Values
            higher than `1` are equivalent to running `groups` independent
            `Conv` operations side-by-side, each having access only to
            `in_channels` // `groups` input channels, and
            concatenating the results along the output channel dimension.
            `in_channels` must be divisible by `groups`.
        - `use_bias`: Whether to add on a bias after the convolution.
        !!! info
            All of `kernel_size`, `stride`, `padding`, `dilation` can be either an
            integer or a sequence of integers. If they are a sequence then the sequence
            should be of length equal to `num_spatial_dims`, and specify the value of
            each property down each spatial dimension in turn.
            If they are an integer then the same kernel size / stride / padding /
            dilation will be used along every spatial dimension.
            `padding` can alternatively also be a sequence of 2-element tuples,
            each representing the padding to apply before and after each spatial
            dimension.
        """
        super().__init__(**kwargs)
        wkey, bkey = jrandom.split(key, 2)

        parse = _ntuple(num_spatial_dims)
        kernel_size = parse(kernel_size)
        stride = parse(stride)
        dilation = parse(dilation)

        if in_channels % groups != 0:
            raise ValueError(
                f"`in_channels` (={in_channels}) must be divisible "
                f"by `groups` (={groups})."
            )

        dtype = get_params_dtype()
        kernel_shape = (out_channels, in_channels // groups) + kernel_size
        self.weight = kernel_init(wkey, kernel_shape, dtype)
        if use_bias:
            bias_shape = (out_channels,) + (1,) * num_spatial_dims
            self.bias = bias_init(bkey, bias_shape, dtype)
        else:
            self.bias = None

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding in ("SAME", "VALID", "CIRCULAR"):
            self.padding = padding
            # size = [(k - 1) * d + 1 for k, d in zip(kernel_size, dilation)]
            # pads = [((k - 1) // 2, k // 2) for k in size]
            # self.circular_pads = [(0, 0), (0, 0)] + pads
            # self.padding = 'VALID'
        elif isinstance(padding, int):
            self.padding = tuple((padding, padding) for _ in range(num_spatial_dims))
        elif isinstance(padding, Sequence) and len(padding) == num_spatial_dims:
            if all(isinstance(element, Sequence) for element in padding):
                self.padding = tuple(padding)
            else:
                self.padding = tuple((p, p) for p in padding)
        else:
            raise ValueError(
                "`padding` must either be SAME, VALID, CIRCULAR, an int or tuple of length "
                f"{num_spatial_dims} containing ints or tuples of length 2."
            )
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

    @jax.named_scope("eqx.nn.Conv")
    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        """**Arguments:**
        - `x`: The input. Should be a JAX array of shape
            `(in_channels, dim_1, ..., dim_N)`, where `N = num_spatial_dims`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        A JAX array of shape `(out_channels, new_dim_1, ..., new_dim_N)`.
        """

        unbatched_rank = self.num_spatial_dims + 1
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank},",
                f" but input has shape {x.shape}.",
            )

        x = x.astype(self.weight.dtype)
        x = jnp.expand_dims(x, axis=0)
        if self.padding == "CIRCULAR":
            size = [(k - 1) * d + 1 for k, d in zip(self.kernel_size, self.dilation)]
            pads = [(0, 0), (0, 0)] + [((k - 1) // 2, k // 2) for k in size]
            x = jnp.pad(x, pads, mode="wrap")
            padding = "VALID"
        else:
            padding = self.padding

        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=self.stride,
            padding=padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, axis=0)
        if self.use_bias:
            x = x + self.bias
        return x


class Conv1d(Conv):
    """As [`equinox.nn.Conv`][] with `num_spatial_dims=1`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = "CIRCULAR",
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        kernel_init: Callable = lecun_normal,
        bias_init: Callable = initializers.zeros,
        *,
        key: Key,
        **kwargs,
    ):
        super().__init__(
            num_spatial_dims=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            key=key,
            **kwargs,
        )


class Conv2d(Conv):
    """As [`equinox.nn.Conv`][] with `num_spatial_dims=2`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = (1, 1),
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = "CIRCULAR",
        dilation: Union[int, Sequence[int]] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        kernel_init: Callable = lecun_normal,
        bias_init: Callable = initializers.zeros,
        *,
        key: Key,
        **kwargs,
    ):
        super().__init__(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            key=key,
            **kwargs,
        )


class Conv3d(Conv):
    """As [`equinox.nn.Conv`][] with `num_spatial_dims=3`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = (1, 1, 1),
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = "CIRCULAR",
        dilation: Union[int, Sequence[int]] = (1, 1, 1),
        groups: int = 1,
        use_bias: bool = True,
        kernel_init: Callable = lecun_normal,
        bias_init: Callable = initializers.zeros,
        *,
        key: Key,
        **kwargs,
    ):
        super().__init__(
            num_spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            key=key,
            **kwargs,
        )


class Sequential(eqx.Module):
    """A sequence of [`equinox.Module`][]s applied in order.
    !!! note
        Activation functions can be added by wrapping them in [`equinox.nn.Lambda`][].
    """

    layers: Tuple[eqx.Module, ...]
    holomorphic: bool = eqx.field(static=True)

    def __init__(self, layers: Sequence[eqx.Module], holomorphic: bool = False):
        self.layers = tuple(layers)
        self.holomorphic = holomorphic

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        """**Arguments:**
        - `x`: Argument passed to the first member of the sequence.
        - `key`: A `jax.random.PRNGKey`, which will be split and passed to every layer
            to provide any desired randomness. (Optional. Keyword only argument.)
        **Returns:**
        The output of the last member of the sequence.
        """

        if key is None:
            keys = [None] * len(self.layers)
        else:
            keys = jrandom.split(key, len(self.layers))
        for layer, key in zip(self.layers, keys):
            x = layer(x, key=key)
        return x

    def __getitem__(self, i: Union[int, slice]) -> eqx.Module:
        if isinstance(i, int):
            return self.layers[i]
        elif isinstance(i, slice):
            return Sequential(self.layers[i])
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")

    def __iter__(self):
        yield from self.layers

    def __len__(self):
        return len(self.layers)


class NoGradLayer(eqx.Module):
    """
    For layers in which the leaves are not considered as differentiable parameters.
    """


def filter_grad(
    fun: Callable, *, has_aux: bool = False, **gradkwargs
) -> Union[Callable, Tuple[Callable, Any]]:
    grad_fn = eqx.filter_grad(fun, has_aux=has_aux, **gradkwargs)
    if has_aux:
        grad_fn, aux = grad_fn

    def filter_grad_fn(*args):
        grad = grad_fn(*args)

        is_nograd = lambda x: isinstance(x, NoGradLayer)
        set_none = lambda x: jtu.tree_map(lambda y: None, x) if is_nograd(x) else x
        grad = jtu.tree_map(set_none, grad, is_leaf=is_nograd)
        return grad
    
    if has_aux:
        return filter_grad_fn, aux
    else:
        return filter_grad_fn


def filter_vjp(
    fun: Callable, *primals, has_aux: bool = False, **vjpkwargs
) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:
    outs = eqx.filter_vjp(fun, *primals, has_aux=has_aux, **vjpkwargs)
    if has_aux:
        out, vjp_fn, aux = outs
    else:
        out, vjp_fn = outs

    def filter_vjp_fn(*args):
        vjp = vjp_fn(*args)

        is_nograd = lambda x: isinstance(x, NoGradLayer)
        set_none = lambda x: jtu.tree_map(lambda y: None, x) if is_nograd(x) else x
        vjp = jtu.tree_map(set_none, vjp, is_leaf=is_nograd)
        return vjp
    
    if has_aux:
        return out, filter_vjp_fn, aux
    else:
        return out, filter_vjp_fn
