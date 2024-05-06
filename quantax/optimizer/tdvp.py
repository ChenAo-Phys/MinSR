from typing import Optional, Callable, Tuple
from functools import partial
import jax
import jax.numpy as jnp

from .solver import auto_pinv_eig, pinvh_solve
from ..state import DenseState, Variational
from ..sampler import Samples
from ..operator import Operator
from ..symmetry import Symmetry
from ..utils import to_array_shard, array_extend, ints_to_array
from ..global_defs import get_default_dtype, is_default_cpl


class TDVP:
    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        use_kazcmarz: bool = False,
    ):
        self._state = state
        self._holomorphic = state._holomorphic
        self._hamiltonian = hamiltonian
        self._imag_time = imag_time
        if solver is None:
            solver = auto_pinv_eig()
        self._solver = solver
        self._use_kazcmarz = use_kazcmarz

        self._energy = None
        self._VarE = None
        self._Omean = None
        self._last_step = None

    @property
    def state(self) -> Variational:
        return self._state

    @property
    def hamiltonian(self) -> Operator:
        return self._hamiltonian

    @property
    def imag_time(self) -> bool:
        return self.imag_time

    @property
    def energy(self) -> Optional[float]:
        return self._energy

    @property
    def VarE(self) -> Optional[float]:
        return self._VarE

    @property
    def vs_type(self) -> int:
        return self._state.vs_type

    def get_Ebar(self, samples: Samples) -> jax.Array:
        Eloc = self._hamiltonian.Oloc(self._state, samples).astype(get_default_dtype())
        # should be pmean here
        Emean = jnp.mean(Eloc * samples.reweight_factor)
        self._energy = Emean.real
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar * samples.reweight_factor).real

        Eloc -= jnp.mean(Eloc)
        Eloc *= jnp.sqrt(samples.reweight_factor / samples.nsamples)
        return Eloc

    def get_Obar(self, samples: Samples) -> jax.Array:
        Omat = self._state.jacobian(samples.spins).astype(get_default_dtype())
        # should be pmean here
        self._Omean = jnp.mean(Omat * samples.reweight_factor[:, None], axis=0)
        Omat -= jnp.mean(Omat, axis=0, keepdims=True)
        Omat *= jnp.sqrt(samples.reweight_factor / samples.nsamples)[:, None]
        return Omat

    def get_step(self, samples: Samples) -> jax.Array:
        Ebar = self.get_Ebar(samples)
        Obar = self.get_Obar(samples)

        if self._use_kazcmarz and self._last_step is not None:
            Ebar -= Obar @ self._last_step
            step = self.solve(Obar, Ebar)
            step += self._last_step
            self._last_step = step
        else:
            step = self.solve(Obar, Ebar)
            
        step = self.build_step(step, self._Omean, self._energy)
        return step

    @partial(jax.jit, static_argnums=0)
    def solve(self, Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        if self.vs_type == 0 and not self._imag_time:
            Ebar *= 1j
        if self.vs_type in (1, 2):
            Obar = jnp.concatenate([Obar.real, Obar.imag], axis=0)
            if self._imag_time:
                Ebar = jnp.concatenate([Ebar.real, Ebar.imag])
            else:
                Ebar = jnp.concatenate([-Ebar.imag, Ebar.real])

        step = self._solver(Obar, Ebar)
        return step

    @partial(jax.jit, static_argnums=0)
    def build_step(self, step: jax.Array, Omean: jax.Array, Emean: float) -> jax.Array:
        if self._imag_time:
            step0 = -jnp.dot(Omean, step)
        else:
            step0 = 1j * Emean - jnp.dot(Omean, step)

        if self.vs_type == 1:
            step = step.reshape(2, -1)
            step = step[0] + 1j * step[1]
        step = step.astype(get_default_dtype())
        step = jnp.insert(step, 0, step0)
        return step


class TDVP_exact(TDVP):
    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        symm: Optional[Symmetry] = None,
    ):
        if solver is None:
            solver = auto_pinv_eig()
        super().__init__(state, hamiltonian, imag_time, solver)

        self._symm = state.symm if symm is None else symm
        basis = self._symm.basis
        basis.make()
        self._spins = ints_to_array(basis.states)
        self._symm_norm = jnp.asarray(basis.get_amp(basis.states))
        if not is_default_cpl():
            self._symm_norm = self._symm_norm.real

    def get_step(self) -> jax.Array:
        wf = self._state(self._spins) / self._symm_norm
        psi = DenseState(wf, self._symm)
        psi_norm = psi.norm()
        psi_norm2 = psi_norm * psi_norm

        H_psi = self._hamiltonian @ psi
        energy = (psi @ H_psi) / psi_norm2
        epsilon = (H_psi - energy * psi) / psi_norm
        self._energy = energy.real
        epsilon = epsilon.wave_function

        Omat = self._state.jacobian(self._spins) * wf[:, None]
        self._Omean = jnp.einsum("s,sk->k", psi.wave_function.conj(), Omat) / psi_norm2
        Omean = jnp.einsum("s,k->sk", psi.wave_function, self._Omean)
        Omat = (Omat - Omean) / psi_norm

        step = self.solve(Omat, epsilon)
        step = self.build_step(step, self._Omean, self._energy)
        return step


@jax.jit
def _AconjB(A: jax.Array, B: jax.Array) -> jax.Array:
    matmul = lambda x, y: x.conj().T @ y
    if A.ndim == 2:
        return matmul(A, B)
    elif A.ndim == 3:
        return jax.vmap(matmul)(A, B)
    else:
        raise NotImplementedError


class TimeEvol(TDVP):
    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        solver: Optional[Callable] = None,
        max_parallel: Optional[int] = None,
    ):
        if solver is None:
            solver = pinvh_solve()
        super().__init__(state, hamiltonian, imag_time=False, solver=solver)
        self._max_parallel = max_parallel

    def get_SF(self, samples: Samples) -> Tuple[jax.Array, jax.Array]:
        if (
            self._max_parallel is None
            or samples.nsamples <= self._max_parallel * jax.local_device_count()
        ):
            Ebar = self.get_Ebar(samples)
            Obar = self.get_Obar(samples)
            Smat = _AconjB(Obar, Obar)
            Fvec = _AconjB(Obar, Ebar)
            return Smat, Fvec
        else:
            return self._get_SF_indirect(samples)

    def _get_SF_indirect(self, samples: Samples) -> Tuple[jax.Array, jax.Array]:
        ndevices = jax.local_device_count()
        Eloc = self._hamiltonian.Oloc(self._state, samples)
        Emean = jnp.mean(Eloc)
        self._energy = Emean.real.item()
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar).real.item()
        Eloc = Eloc.reshape(ndevices, -1)
        Eloc = array_extend(Eloc, self._max_parallel, axis=1)
        nsplits = Eloc.shape[1] // self._max_parallel
        Eloc = jnp.split(Eloc, nsplits, axis=1)

        nsamples, nsites = samples.spins.shape
        spins = samples.spins.reshape(ndevices, -1, nsites)
        spins = array_extend(spins, self._max_parallel, 1, padding_values=1)
        spins = jnp.split(spins, nsplits, axis=1)

        nparams = self._state.nparams
        dtype = get_default_dtype()
        Smat = to_array_shard(jnp.zeros((ndevices, nparams, nparams), dtype))
        Fvec = to_array_shard(jnp.zeros((ndevices, nparams), dtype))
        Omean = to_array_shard(jnp.zeros((ndevices, nparams), dtype))
        for s, e in zip(spins, Eloc):
            Omat = self._state.jacobian(s.reshape(-1, nsites))
            Omat = Omat.reshape(ndevices, -1, nparams).astype(dtype)
            Omean += jnp.sum(Omat, axis=1)
            newS = _AconjB(Omat, Omat)
            newF = _AconjB(Omat, e)
            Smat += newS
            Fvec += newF
        # psum here, nsamples definition should be modified to all samples across nodes
        Smat = jnp.sum(Smat, axis=0) / nsamples
        Fvec = jnp.sum(Fvec, axis=0) / nsamples
        Omean = jnp.sum(Omean, axis=0) / nsamples
        self._Omean = Omean

        Smat = Smat - jnp.outer(Omean.conj(), Omean)
        Fvec = Fvec - Omean.conj() * Emean
        return Smat, Fvec

    def get_step(self, samples: Samples) -> jax.Array:
        if not jnp.allclose(samples.reweight_factor, 1.0):
            raise ValueError("TimeEvol is only for non-reweighted samples")

        Smat, Fvec = self.get_SF(samples)
        step = self.solve(Smat, Fvec)
        step = self.build_step(step, self._Omean, self._energy)
        return step

    def solve(self, Smat: jax.Array, Fvec: jax.Array) -> jax.Array:
        if self.vs_type == 0:
            Fvec *= 1j
        else:
            Smat = Smat.real
            Fvec = -Fvec.imag
        step = self._solver(Smat, Fvec)
        return step


class SGD(TDVP):
    def __init__(self, state: Variational, hamiltonian: Operator):
        super().__init__(state, hamiltonian, imag_time=True, solver=None)

    def get_step(self, samples: Samples) -> jax.Array:
        """Requires revision"""
        # Eloc = self._hamiltonian.Oloc(self._state, samples)
        # vec = (Eloc - self._energy).conj()
        # vec *= samples.reweight_factor / samples.wave_function
        # step = self._state.vjp(samples.spins, vec).conj() / samples.nsamples
        # step0 = -jnp.mean(self._state.jvp(samples.spins, step))
        # step = jnp.insert(step, 0, step0)
        # return step
