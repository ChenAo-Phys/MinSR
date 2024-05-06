# MinSR
Repository for the paper

_Empowering deep neural quantum states through efficient optimization_

Related links:
- Article (not yet published)
- [Preprint](https://arxiv.org/abs/2302.01941) (not up-to-date)
- [GitHub](https://github.com/ChenAo-Phys/MinSR)
- [Zenodo](https://zenodo.org/doi/10.5281/zenodo.7657551)

![geometry](images/geometry_Fig1a.pdf)

![matrix](images/matrix_Fig1b.pdf)


## Contents

- Images
- Data
- Quantax, a package for reproducing results in the paper
- Tutorials for Quantax
- Exemplary codes to train networks and measure the variational energy
- Some well-trained network weights

## Quantax

### Installation

#### 1. Create a conda environment with QuSpin
Quantax relies on the dev_0.3.8 branch of QuSpin, which can't be easily installed
through pip or conda. Follow this [instruction](https://github.com/QuSpin/QuSpin/discussions/665) for manual installation.

#### 2. Install Quantax
Clone the repository `git clone https://github.com/ChenAo-Phys/MinSR.git`

Enter the foler `cd MinSR`

Install Quantax `pip install .`


### Supported platforms
- CPU
- Nvidia GPU


### Parallelism
- Quantax can work only if all devices are on the same host
- Multiple-host compatibility is to be implemented
