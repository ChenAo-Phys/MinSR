from __future__ import annotations
from typing import Callable, Optional, Tuple, Union, Sequence, BinaryIO
from numpy.typing import ArrayLike
from jaxtyping import PyTree
from pathlib import Path

from warnings import warn
import numpy as np
import math
import jax
import jax.numpy as jnp
import jax.flatten_util as jfu
import equinox as eqx

from .state import State
from ..symmetry import Symmetry
from ..nn import NoGradLayer, filter_grad, Theta0Layer
from ..utils import (
    is_sharded_array,
    to_array_shard,
    filter_replicate,
    array_extend,
    tree_fully_flatten,
)
from ..global_defs import get_params_dtype, is_params_cpl


_Array = Union[np.ndarray, jax.Array]


class Variational(State):
    """
    Variational state.

    self.vs_type:
        0: real parameters -> real outputs or (holomorphic complex -> complex)
        1: non-holomorphic complex parameters -> complex outputs
            ∇ψ(θ) = [∇(ψr)(θr) + 1j * ∇(ψi)(θr), ∇(ψr)(θi) + 1j * ∇(ψi)(θi)]
        2: real parameters -> complex outputs
            ∇ψ(θ) = ∇(ψr)(θ) + 1j * ∇(ψi)(θ)
    """

    def __init__(
        self,
        ansatz: Union[eqx.Module, Sequence[eqx.Module]],
        param_file: Optional[str] = None,
        symm: Optional[Symmetry] = None,
        max_parallel: Optional[int] = None,
        input_fn: Optional[Callable] = None,
        output_fn: Optional[Callable] = None,
    ):
        super().__init__(symm)
        if isinstance(ansatz, eqx.Module):
            ansatz = (ansatz,)
        else:
            ansatz = tuple(ansatz)
        self._ansatz = ansatz

        holomorphic = [a.holomorphic for a in ansatz if hasattr(a, "holomorphic")]
        self._holomorphic = len(holomorphic) > 0 and all(holomorphic)

        # load params
        if param_file is not None:
            ansatz = eqx.tree_deserialise_leaves(param_file, ansatz)
        self._ansatz = filter_replicate(ansatz)
        self._max_parallel = max_parallel
        # the number of allowed forward inputs per device
        self._max_eval = None if max_parallel is None else max_parallel // self.nsymm

        # initialize forward and backward
        self._init_forward(input_fn, output_fn)
        self._init_backward()
        self._maximum = None

        # for params flatten and unflatten
        params, others = self.partition()
        params, self._unravel_fn = jfu.ravel_pytree(params)
        self._nparams = params.size
        if params.dtype == jnp.float16:
            self._params_copy = params.astype(jnp.float32)
        else:
            self._params_copy = None

        batch = jnp.ones((1, self.nsites), jnp.int8)
        outputs = jax.eval_shape(self.forward_vmap, batch)
        is_p_cpl = is_params_cpl()
        is_outputs_cpl = np.issubdtype(outputs, np.complexfloating)
        if (not is_p_cpl) and (not is_outputs_cpl):
            self._vs_type = 0
        elif is_p_cpl and is_outputs_cpl and self.holomorphic:
            self._vs_type = 0
        elif is_p_cpl and is_outputs_cpl:
            self._vs_type = 1
        elif (not is_p_cpl) and is_outputs_cpl:
            self._vs_type = 2
        else:
            raise RuntimeError("Parameter or output datatype not supported.")

    @property
    def ansatz(self) -> eqx.Module:
        return self._ansatz

    @property
    def holomorphic(self) -> bool:
        return self._holomorphic

    @property
    def max_parallel(self) -> int:
        return self._max_parallel

    @property
    def nparams(self) -> int:
        return self._nparams

    @property
    def vs_type(self) -> int:
        return self._vs_type

    def _init_forward(
        self, input_fn: Optional[Callable] = None, output_fn: Optional[Callable] = None
    ) -> None:
        if len(self.ansatz) > 1 and (input_fn is None or output_fn is None):
            raise ValueError(
                "The default input_fn and output_fn only works for single ansatz."
            )
        if input_fn is None:
            input_fn = lambda s: [self.symm.get_symm_spins(s)]
        if output_fn is None:
            output_fn = lambda x: self.symm.symmetrize(x[0])

        self.input_fn = input_fn
        self.output_fn = output_fn

        def forward_fn(ansatz: eqx.Module, spin: jax.Array) -> jax.Array:
            inputs = input_fn(spin)
            outputs = []
            for s, f in zip(inputs, ansatz):
                batch = s.shape[:-1]
                psi = jax.vmap(f)(s.reshape(-1, s.shape[-1]))
                outputs.append(psi.reshape(batch))
            # print(self, len(outputs))
            outputs = output_fn(outputs)
            return outputs

        self.forward_fn = forward_fn
        forward_vmap = eqx.filter_jit(jax.vmap(forward_fn, in_axes=(None, 0)))
        self.forward_vmap = lambda spins: forward_vmap(self.ansatz, spins)

    def _init_backward(self) -> None:
        """
        Generate functions for computing 1/ψ dψ/dθ
        """

        def forward(
            params: jax.Array, others: eqx.Module, spin: jax.Array
        ) -> jax.Array:
            if self.vs_type == 1:
                params = params.reshape(2, -1)
                params = params[0] + 1j * params[1]
            params = self.get_params_unflatten(params)
            ansatz = self.combine(params, others)
            psi = self.forward_fn(ansatz, spin)
            return psi / jax.lax.stop_gradient(psi)

        def grad_fn(ansatz: eqx.Module, spin: jax.Array) -> jax.Array:
            params, others = self.partition(ansatz)
            params = tree_fully_flatten(params)
            if self.vs_type == 1:
                params = jnp.concatenate([params.real, params.imag])
            if self.vs_type == 0:
                grad_fn = filter_grad(forward, holomorphic=self.holomorphic)
                grad = grad_fn(params, others, spin)
            else:
                gradfn_real = filter_grad(lambda *args: forward(*args).real)
                gradfn_imag = filter_grad(lambda *args: forward(*args).imag)
                gr = gradfn_real(params, others, spin)
                gi = gradfn_imag(params, others, spin)
                grad = gr + 1j * gi
            return grad

        self.grad_fn = grad_fn
        self.grad = lambda s: eqx.filter_jit(grad_fn)(self.ansatz, s)
        grad_vmap = jax.vmap(grad_fn, in_axes=(None, 0))
        self.jacobian = lambda spins: eqx.filter_jit(grad_vmap)(self.ansatz, spins)

    def __call__(
        self, fock_states: _Array, ref_states: Optional[_Array] = None
    ) -> jax.Array:
        ndevices = jax.local_device_count()
        nsamples, nsites = fock_states.shape
        is_input_sharded = is_sharded_array(fock_states)

        if not is_input_sharded:
            fock_states = array_extend(fock_states, ndevices, axis=0, padding_values=1)
        if self._max_eval is None or nsamples <= ndevices * self._max_eval:
            fock_states = to_array_shard(fock_states)
            psi = self.forward_vmap(fock_states)
        else:
            fock_states = fock_states.reshape(ndevices, -1, nsites)
            nsamples_per_device = fock_states.shape[1]
            fock_states = array_extend(fock_states, self._max_eval, 1, 1)

            nsplits = fock_states.shape[1] // self._max_eval
            if isinstance(fock_states, jax.Array):
                fock_states = to_array_shard(fock_states)
                fock_states = jnp.split(fock_states, nsplits, axis=1)
                psi = [self.forward_vmap(s.reshape(-1, nsites)) for s in fock_states]
            else:
                fock_states = np.split(fock_states, nsplits, axis=1)
                fock_states = [s.reshape(-1, nsites) for s in fock_states]
                psi = [self.forward_vmap(to_array_shard(s)) for s in fock_states]

            psi = jnp.concatenate([p.reshape(ndevices, -1) for p in psi], axis=1)
            psi = psi[:, :nsamples_per_device].flatten()

        if not is_input_sharded:
            psi = filter_replicate(psi[:nsamples])
            maximum = jnp.max(jnp.abs(psi))
            if self._maximum is None or maximum > self._maximum:
                self._maximum = maximum
        return psi

    def partition(
        self, ansatz: Optional[eqx.Module] = None
    ) -> Tuple[eqx.Module, eqx.Module]:
        if ansatz is None:
            ansatz = self._ansatz
        is_nograd = lambda x: isinstance(x, NoGradLayer)
        return eqx.partition(ansatz, eqx.is_inexact_array, is_leaf=is_nograd)

    def combine(self, params: eqx.Module, others: eqx.Module) -> eqx.Module:
        is_nograd = lambda x: isinstance(x, NoGradLayer)
        return eqx.combine(params, others, is_leaf=is_nograd)

    def get_params_flatten(self) -> jax.Array:
        params, others = self.partition()
        return tree_fully_flatten(params)

    def get_params_unflatten(self, params: jax.Array) -> PyTree:
        return filter_replicate(self._unravel_fn(params))

    def update_theta0(self, step0: ArrayLike, rescale: bool = True) -> None:
        if (
            rescale
            and self._maximum is not None
            and not math.isclose(self._maximum, 0.0)
            and math.isfinite(self._maximum)
        ):
            step0 += jnp.log(self._maximum)

        self._maximum = None
        if not jnp.isfinite(step0):
            warn(f"Got invalid theta0 update {step0}.")
            return

        is_theta0 = lambda leaf: isinstance(leaf, Theta0Layer)
        theta0_layer, others = eqx.partition(self.ansatz, is_theta0, is_leaf=is_theta0)
        theta0, unravel_fn = jfu.ravel_pytree(theta0_layer)
        theta0_layer = unravel_fn(theta0 - step0)
        self._ansatz = eqx.combine(theta0_layer, others, is_leaf=is_theta0)

    def update(self, step: jax.Array, rescale: bool = True) -> None:
        self.update_theta0(step[0], rescale)

        step = -step[1:]
        if not jnp.all(jnp.isfinite(step)):
            num_err = step.size - jnp.sum(jnp.isfinite(step))
            warn(
                f"Got invalid update step with {step.size} elements and {num_err} NaN"
                "or Inf values. The update is interrupted."
            )
            return
        if not is_params_cpl():
            step = step.real

        dtype = get_params_dtype()
        if dtype != jnp.float16:
            step = step.astype(dtype)
            step = self.get_params_unflatten(step)
            self._ansatz = eqx.apply_updates(self._ansatz, step)
        else:
            self._params_copy += step.astype(jnp.float32)
            new_params = self.get_params_unflatten(self._params_copy)
            params, others = self.partition()
            self._ansatz = self.combine(new_params, others)

    def rescale(self) -> None:
        self.update_theta0(0.0, rescale=True)

    def save(self, file: Union[str, Path, BinaryIO]) -> None:
        eqx.tree_serialise_leaves(file, self._ansatz)

    def __add__(self, other: Callable) -> Variational:
        if not isinstance(Variational):
            other = Functional(other)
        input_fn = lambda s: [*self.input_fn(s), *other.input_fn(s)]
        sep = len(self.ansatz)
        output_fn = lambda x: self.output_fn(x[:sep]) + other.output_fn(x[sep:])
        return CombineVs([self, other], input_fn, output_fn)

    def __radd__(self, other: Callable) -> Variational:
        return self + other

    def __mul__(self, other: Callable) -> Variational:
        if not isinstance(other, Variational):
            other = Functional(other)
        input_fn = lambda s: [*self.input_fn(s), *other.input_fn(s)]
        sep = len(self.ansatz)
        output_fn = lambda x: self.output_fn(x[:sep]) * other.output_fn(x[sep:])
        return CombineVs([self, other], input_fn, output_fn)

    def __rmul__(self, other: Callable) -> Variational:
        return self * other

    def to_flax_model(self, package="netket", real_outputs: bool = False):
        """
        Convert the state to a flax model compatible with other packages.
        Training the model in other packages may be unstable.
        The supported packages are listed below
            netket (default), input 1/-1, output log(psi)
            jvmc, input 1/0, output log(psi)
        """
        params, others = self.partition()
        params, unravel_fn = jfu.ravel_pytree(params)

        class Model:
            def init(self, *args):
                return {"params": {"params": params}}

            @staticmethod
            def apply(params: dict, inputs: jax.Array, **kwargs) -> jax.Array:
                if package == "jvmc":
                    inputs = 2 * inputs - 1
                params = unravel_fn(params["params"]["params"])
                ansatz = eqx.combine(params, others)
                psi = jax.vmap(self.forward_fn, in_axes=(None, 0))(ansatz, inputs)
                if real_outputs:
                    if jnp.iscomplexobj(psi):
                        raise RuntimeError(
                            "The outputs are specified to be real, but got complex"
                        )
                else:
                    psi += 0j
                return jnp.log(psi)

        return Model()


def Functional(fn: Callable) -> Variational:
    if isinstance(fn, eqx.Module):
        fn = fn.__call__  # remove parameter dependence in fn
    fn = eqx.nn.Lambda(fn)
    return Variational(fn)


def CombineVs(
    states: Sequence[Variational],
    input_fn: Callable[[jax.Array], Tuple[jax.Array]],
    output_fn: Callable[[Tuple[jax.Array]], jax.Array],
) -> Variational:
    ansatz = sum([vs.ansatz for vs in states], start=())

    max_parallel = None
    for vs in states:
        if vs.max_parallel is not None:
            if max_parallel is None or vs.max_parallel < max_parallel:
                max_parallel = vs.max_parallel

    variational = Variational(
        ansatz, None, states[0].symm, max_parallel, input_fn, output_fn
    )
    return variational