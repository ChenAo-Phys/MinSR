from typing import Optional, Callable, Tuple, List, Any
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from .tdvp import TDVP
from .solver import minsr_pinv_eig
from ..state import Variational
from ..nn import Sequential, filter_vjp
from ..sampler import Samples
from ..operator import Operator
from ..utils import tree_fully_flatten, to_array_shard, array_extend
from ..global_defs import get_default_dtype


class MinSR(TDVP):
    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        solver: Optional[Callable] = None,
    ):

        if solver is None:
            solver = minsr_pinv_eig()
        super().__init__(state, hamiltonian, imag_time=True, solver=solver)

        if self.vs_type == 1:
            raise ValueError("'MinSR' optimizer doesn't support vs_type 1")

        self._nodes = []
        params, others = state.partition(state.ansatz)
        for params_i in params:
            nparams = tree_fully_flatten(params_i).size
            if nparams == 0:
                self._nodes.append("unparametrized")
            elif not isinstance(params_i, Sequential):
                self._nodes.append("not_sequential")
            else:
                idx_parametrized = []
                for i, layer in enumerate(params_i):
                    vals = tree_fully_flatten(layer)
                    if vals.size > 0:
                        idx_parametrized.append(i)

                block_len = np.sqrt(len(idx_parametrized)).astype(int)
                new_nodes = []
                for i, l in enumerate(idx_parametrized):
                    if i % block_len == block_len - 1 or i == len(idx_parametrized) - 1:
                        new_nodes.append(l)
                self._nodes.append(new_nodes)

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0))
    def _forward(
        self, ansatz: Tuple[eqx.Module, ...], spin: jax.Array
    ) -> Tuple[List[jax.Array], jax.Array]:
        inputs = self.state.input_fn(spin)
        neurons = []
        batch = []
        outputs = []
        remaining_layers = []

        for net, x, nodes in zip(ansatz, inputs, self._nodes):
            batch.append(x.shape[:-1])
            x = x.reshape(-1, x.shape[-1])
            new_neurons = [x]

            if not isinstance(nodes, list):
                neurons.append(new_neurons)
                outputs.append(jax.vmap(net)(x))
                remaining_layers.append(lambda x: x)
            else:
                for i, layer in enumerate(net):
                    x = jax.vmap(layer)(x)
                    if i == nodes[-1]:
                        if self.vs_type == 2 and jnp.iscomplexobj(x):
                            raise NotImplementedError(
                                "'MinSR' optimizer only supports vs_type 2 with "
                                "complex neurons in the last few unparametrized layers."
                            )
                        outputs.append(x)
                        break
                    if i in nodes:
                        new_neurons.append(x)
                neurons.append(new_neurons)
                remains = jax.vmap(net[i + 1 :]) if i < len(net) - 1 else lambda x: x
                remaining_layers.append(remains)

        def output_fn(x: List[jax.Array]) -> jax.Array:
            x = [l(xi).reshape(b) for l, xi, b in zip(remaining_layers, x, batch)]
            psi = self.state.output_fn(x)
            return psi / jax.lax.stop_gradient(psi)

        if self.vs_type == 0:
            jac_out = jax.grad(output_fn, holomorphic=self.state.holomorphic)
            deltas = jac_out(outputs)
        else:
            jac_real = jax.grad(lambda x: output_fn(x).real)(outputs)
            jac_imag = jax.grad(lambda x: output_fn(x).imag)(outputs)
            deltas = [re + 1j * im for re, im in zip(jac_real, jac_imag)]
        return neurons, deltas

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def _layer_backward(
        self, layers: eqx.Module, neuron: jax.Array, delta: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        forward = lambda net, x: net(x)

        @partial(jax.vmap, in_axes=(None, 0, 0))
        def backward(net, x, delta):
            f_vjp = filter_vjp(forward, net, x)[1]
            vjp_vals, delta = f_vjp(delta)
            return tree_fully_flatten(vjp_vals), delta

        if self.vs_type == 0:
            grad, delta = backward(layers, neuron, delta)
        else:
            gr, delta_real = backward(layers, neuron, delta.real)
            gi, delta_imag = backward(layers, neuron, delta.imag)
            grad = gr + 1j * gi
            delta = delta_real + 1j * delta_imag if delta_real is not None else None
        grad = jnp.sum(grad.astype(get_default_dtype()), axis=0)
        return grad, delta

    @eqx.filter_jit
    def _reversed_scan_layers(
        self,
        ansatz: Tuple[eqx.Module, ...],
        spins: jax.Array,
        fn_on_jac: Callable,
        init_vals: Any,
    ):
        neurons, deltas = self._forward(ansatz, spins)

        for net, neurons_i, delta, nodes in zip(
            reversed(ansatz), reversed(neurons), reversed(deltas), reversed(self._nodes)
        ):
            if nodes == "unparametrized":
                continue
            elif nodes == "not_sequential":
                jac, delta = self._layer_backward(net, neurons_i[0], delta)
                if jac.size > 0:
                    init_vals = fn_on_jac(jac, init_vals)
            else:
                end = nodes[-1] + 1
                nodes = [0] + [n + 1 for n in nodes[:-1]]
                for start, neuron in zip(reversed(nodes), reversed(neurons_i)):
                    layers = net[start:end]
                    end = start
                    jac, delta = self._layer_backward(layers, neuron, delta)
                    if jac.size > 0:
                        init_vals = fn_on_jac(jac, init_vals)

        return init_vals

    @partial(jax.jit, static_argnums=0, donate_argnums=1)
    def _get_Obar(self, Omat: jax.Array, reweight: jax.Array) -> jax.Array:
        # should be pmean here
        Omat -= jnp.mean(Omat, axis=0, keepdims=True)
        Omat *= jnp.sqrt(reweight[:, None] / Omat.shape[0])
        Omat = array_extend(Omat, jax.local_device_count(), axis=1)
        Omat = to_array_shard(Omat, sharded_axis=1)
        if self.vs_type != 0:
            Omat = jnp.concatenate([Omat.real, Omat.imag], axis=0)
        return Omat
    
    @partial(jax.jit, static_argnums=0, donate_argnums=1)
    def _Tmat_scan_fn(
        self, jac: jax.Array, vals: Tuple[jax.Array, List, jax.Array]
    ) -> Tuple[jax.Array, List, jax.Array]:
        Tmat, Omean, reweight = vals
        Omean.append(jnp.mean(jac * reweight[:, None], axis=0))
        Obar = self._get_Obar(jac, reweight)
        Tmat += Obar @ Obar.conj().T
        return Tmat, Omean, reweight

    def get_Tmat(self, samples: Samples) -> jax.Array:
        Ns = samples.nsamples if self.vs_type == 0 else 2 * samples.nsamples
        if self.vs_type == 0:
            dtype = get_default_dtype()
        else:
            if get_default_dtype() in (jnp.float64, jnp.complex128):
                dtype = jnp.float64
            else:
                dtype = jnp.float32
        Tmat = jnp.zeros((Ns, Ns), dtype=dtype)

        Omean = []
        init_vals = (Tmat, Omean, samples.reweight_factor)
        Tmat, Omean, reweight = self._reversed_scan_layers(
            self.state.ansatz, samples.spins, self._Tmat_scan_fn, init_vals
        )
        self._Omean = jnp.concatenate(Omean[::-1])
        return Tmat
    
    @partial(jax.jit, static_argnums=0, donate_argnums=1)
    def _Ohvp_scan_fn(
        self, jac: jax.Array, vals: Tuple[List, jax.Array, jax.Array]
    ) -> Tuple[List, jax.Array, jax.Array]:
        outputs, reweight, vec = vals
        nparams = jac.shape[1]
        Obar = self._get_Obar(jac, reweight)
        Ohvp = (Obar.conj().T @ vec)[:nparams]
        outputs.append(Ohvp)
        return outputs, reweight, vec

    def Ohvp(self, samples: Samples, vec: jax.Array) -> jax.Array:
        """
        Compute O^dag @ vec. vec @ jac is used instead of vjp for better precision.
        """
        init_vals = ([], samples.reweight_factor, vec)
        outputs, reweight, vec = self._reversed_scan_layers(
            self.state.ansatz, samples.spins, self._Ohvp_scan_fn, init_vals
        )
        outputs = jnp.concatenate(outputs[::-1])
        return outputs

    def get_step(self, samples: Samples) -> jax.Array:
        Ebar = self.get_Ebar(samples)
        Tmat = self.get_Tmat(samples)
        step = self.solve(samples, Tmat, Ebar)
        step = self.build_step(step, self._Omean, self._energy)
        return step

    def solve(self, samples: Samples, Tmat: jax.Array, Ebar: jax.Array) -> jax.Array:
        if self.vs_type != 0:
            Ebar = jnp.concatenate([Ebar.real, Ebar.imag])
        Tinv_E = self._solver(Tmat, Ebar)
        if isinstance(Tinv_E, tuple):
            Tinv_E, self._solver_info = Tinv_E
        step = self.Ohvp(samples, Tinv_E)
        return step
