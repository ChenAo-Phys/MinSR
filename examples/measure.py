"""
Choose the network from
1: 16 layers, 16 channels every layer, 3x3 kernels 34944 parameters in total
2: 16 layers, 32 channels every layer, 3x3 kernels, 139008 parameters in total
3: 30 layers, 64 channels every layer, 3x3 kernels, 1071488 parameters in total

The variational energies of these networks correspond to the results of ResNet2(MinSR) 
in Fig. 2b
"""

network_idx = 3

import pathlib
import quantax as qtx
from quantax.symmetry import TotalSz, SpinInverse, C4v

lattice = qtx.sites.Square(10)
N = lattice.nsites

H = qtx.operator.Heisenberg(J=[1, 0.5], n_neighbor=[1, 2], msr=True)

path = pathlib.Path(__file__).parent.parent
if network_idx == 1:
    net = qtx.nn.ResSum(16, 16, 3, use_sinh=True)
    param_file = path.joinpath("params", "square10x10_1616.eqx")
elif network_idx == 2:
    net = qtx.nn.ResSum(16, 32, 3, use_sinh=True)
    param_file = path.joinpath("params", "square10x10_1632.eqx")
elif network_idx == 3:
    net = qtx.nn.ResSum(30, 64, 3, use_sinh=True)
    param_file = path.joinpath("params", "square10x10_3064.eqx")
else:
    raise ValueError("Wrong network index.")

state = qtx.state.Variational(
    net,
    param_file=param_file,
    symm=TotalSz() + SpinInverse() + C4v(),
    max_parallel=32768,  # Reduce this number in case of out-of-memory error
)

print("Initializing the sampler. This may take a while...")
sampler = qtx.sampler.NeighborExchange(
    state, 10000, thermal_steps=100 * N, sweep_steps=10 * N, n_neighbor=[1, 2]
)

energy_data = qtx.utils.DataTracer()
VarE_data = qtx.utils.DataTracer()

for i in range(100):
    samples = sampler.sweep()
    energy, VarE = H.expectation(state, samples, return_var=True)
    energy_data.append(energy)
    VarE_data.append(VarE)
    var = VarE_data.mean() - energy_data.mean() ** 2
    print(f"Iteration {i} \t energy {energy_data.mean()} \t Var(E) {var}")
