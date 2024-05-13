import matplotlib.pyplot as plt
import quantax as qtx
from quantax.symmetry import TotalSz

lattice = qtx.sites.Square(10)
N = lattice.nsites

H = qtx.operator.Heisenberg(J=[1, 0.5], n_neighbor=[1, 2], msr=True)

net = qtx.nn.ResSum(16, 16, 3, use_sinh=True)
state = qtx.state.Variational(
    net,
    symm=TotalSz(),
    max_parallel=32768,  # Reduce this number in case of out-of-memory error
)

# reweighted sampling for more efficient training (only for N <= 100)
sampler = qtx.sampler.NeighborExchange(state, 10000, reweight=1.0, n_neighbor=[1, 2])
optimizer = qtx.optimizer.MinSR(state, H)

energy = qtx.utils.DataTracer()
VarE = qtx.utils.DataTracer()

for n in range(20000):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)
    state.update(step * 1e-3)

    energy.append(optimizer.energy)
    VarE.append(optimizer.VarE)
    if n % 10 == 9:
        energy.plot(batch=10, start=-1000, baseline=-199.0859)
        plt.savefig("energy.pdf")
        plt.clf()
        VarE.plot(batch=10, start=-1000)
        plt.savefig("VarE.pdf")
        plt.clf()
    if n % 100 == 99:
        state.save(f"params_{n+1}.eqx")
