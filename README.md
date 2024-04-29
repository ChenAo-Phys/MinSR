# MinSR
Codes for reproducing the results in MinSR

## Contents

- Quantax, a package to reproduce results in MinSR
- Tutorials for Quantax
- Exemplary codes to train networks and measure the variational energy
- Some well-trained network weights

## Installation

### 1. Create a conda environment with QuSpin
quantax relies on the dev_0.3.8 branch of QuSpin, which can't be easily installed
through pip or conda. Follow this [instruction](https://github.com/QuSpin/QuSpin/discussions/665) for manual installation.

### 2. Install Quantax
Open a terminal, `cd` into the downloaded MinSR folder, and `python setup.py install`


## Supported platforms
- CPU
- Nvidia GPU


## Parallelism
- Quantax can work only if all devices are on the same host
- Multiple-host compatibility is to be implemented
