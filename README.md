
# Applications of MLMC in Proton Beam Therapy (PBT)

This repository contains the code used to generate results in the following master's thesis:

**Thesis Title:** Stochastic Modelling of Proton Transport and Spatial Therapeutic Dose Prediction with Multilevel Monte Carlo (MLMC)  
**Author:** Katie Zoltan  
**Institution:** University of Oxford 
**Research Group:** Numerical Analysis Group, Mathematical Institute  
**Supervisor(s):** Professor Mike Giles FRS  
**Academic Year:** 2025-26
**Course:** MSc in Mathematical Modelling and Scientific Computing

## Overview and Objectives

The function of this code is to estimate the absorbed dose of a monoenergetic proton beam in a water phantom using Multilevel Monte Carlo. To do this, we compared, adapted and evaluated:

- (1) the underlying proton transport model and its numerical treatment during simulation
- (2) the dose computation technique used to convert a simulated energetic proton and its track into a dose map in 2D or 3D space
- (3) the Monte Carlo method used estimate the overall dose map resulting from millions of proton simulations   

Firstly, we evaluate a pair of SDE proton transport models, the first of which originates from _Chronholm and Pryer, 2026_ (see _References_) simulated with the Euler-Maruyama discretisation (specifically, a geometry preserving scheme using the Riemannian exponential map on the sphere) where one is parameterised in energy and the other in track length. Secondly, the compare the existing dose computation method in _Chronholm and Pryer_ with a smoother alternative using finite element methods, which is developed in the adjoining thesis to this work. Thirdly, we compare the use of these methods with standard Monte Carlo against Multilevel Monte Carlo for this task, alongside suggesting techniques for ensuring dose positivity. 

Overall, the main contribution of this work was that, using the new simulation and dose computation method that we developed, the MLMC was successful at achieving significantly improved computational complexity over standard Monte Carlo, in the homogeneous medium tested. Future work would aim to extend this work to include other media, inhomogeneity, more complex geometric structures, and explore other variants of MLMC such as antithetic MLMC. Further details are given in the written work, for which a URL will be provided in due course.

## Installation

### HPC Hardware

Due to the high objective computational complexity of this task, access to a High-Performance Computing (HPC) cluster is recommended for running large-scale experiments. For example, running MLMC with root mean-square-error $\eps < 10^{-1}$ is likely to require at least $\mathcal{O}(10^{7})$ proton simulations in 2D, based on our use of this code. In 3D, we expect the complexity to worsen significantly for all methods with the current setup. 

The experiments presented in this thesis were performed using the an HPC cluster owned by the University of Oxford. 128 CPU cores were used over which proton simulations were run in parallel. No job scheduler was used, and Linux OS was used. One full dose map under the conditions described above in 2D was produced in the order of hours. Note that improving the error requirement can notably increase the runtime.   

Smaller plots (e.g. large $\eps \sim \\mathcal{O}(1)$, Monte Carlo dose maps using $\mathcal{O}(10^{5})$ simulations) and unit tests may be executed locally. 

### Software Requirements

The following software is required:

- Python [version requirements not yet determined]
- All standard scientific computing libraries, plus Numba [version requirements not yet determined]

### Installation from Source

The project is intended to be distributed as an installable Python package, with its dependencies managed through [`pyproject.toml`](pyproject.toml). Currently, the project can be installed directly from the repository.

Clone the repository:

```bash
git clone https://github.com/kezoltan/MSc-Thesis-Proton-Therapy-Monte-Carlo-.git
cd MSc-Thesis-Proton-Therapy-Monte-Carlo-
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project and its dependencies:

```bash
pip install .
```

For development, the package can be installed in editable mode:

```bash
pip install -e .
```

### Future Package Distribution

The project is intended to be published on the Python Package Index (PyPI), allowing users to install the package and its dependencies directly. 

*Note: PyPI installation is not yet available. We will update this section in due coarse.*

## Usage

After installation, the entire project can be executed using:

```bash
python src/dose_main.py
```

### Parameterisation

All parameters can be set inside of src/doseparams.py. We provide default parameters for most variables for the purposes of testing, however we expect to provide alternative presets (e.g. for different materials) in future. Currently, the user should pay attention to:

| Core Parameter Name | Function | Suggested Default* |
|----------|----------|
| sde_model | Select which proton SDE is used | 'energy' |
| dose_method | Select which dose computation technique is used | 'SF' |
| sampling_type |  Select which Monte Carlo method is used | 'mlmc' |
| file_path | Select where data and pictures are saved | N/A |
| SPATIAL_DIM | Select the number of spatial dimensions in which to estimate | 2 |
| side_len | Uniform distance between two adjacent nodes along the same coordinate direction | 0.02-0.1 (cm) |   
| E0 | Initial beam energy | 50-70 (MeV) |
| Eps | List of rmse values for use in MLMC | $\geq 0.2$ |

*The suggested defaults above are based on the results of experiments but are not restrictive. The first three parameter settings produced the best overall computational complexity (when combined) and hence are recommended, but these results have yet to be reproduced. If reproducing the thesis results, other values of the first three settings will be required for comparison and no other parameters (excluding the file_path) should be modified.

## Testing and Validation

Reproducible regimes for testing are currently being implemented. As such, **we do not recommend installing this work for any reason other than reproducing results at this time.**

## References

The following publications and resources informed the development of this project:

1. [Chronholm V and Pryer T], *Geometry, Energy and Sensitivity in Stochastic Proton Dynamics*, 2026, DOI: 10.48550/arXiv.2509.13223
2. [Giles M], *Multilevel Monte Carlo Path Simulation*, Operations Research, 2008, DOI: 10.1287/opre.1070.0496 URL: https://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf

Please also see the references listed within the adjoining MSc thesis. 

### Citing This Project

If you use this repository in academic work, please cite:

> Katherine Zoltan and Mike Giles, *Stochastic Modelling of Proton Transport and Spatial Therapeutic Dose Prediction with MLMC*, MSc Thesis, University of Oxford, 2026

DOI or journal publication does not yet exist. 

## Contributors

This project was developed by Katie Zoltan and Professor Mike Giles FRS. Our main contributors are Peiren Wang and Alice Watts.

See [CONTRIBUTING] for further information.

## License

This project is distributed under the MIT License. See [LICENSE] for details.

## Contact

For questions regarding this project, please contact:
  
Katie Zoltan (https://github.com/kezoltan)
katherine.zoltan@bnc.ox.ac.uk