# MLMC for Proton Beam Therapy (PBT) Dose Estimation

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

Firstly, we evaluate a pair of SDE proton transport models, the first of which originates from _Chronholm and Pryer, 2026_ (see _References_) simulated with the Euler-Maruyama discretisation (specifically, a geometry preserving scheme using the Riemannian exponential map on the sphere) where one is parameterised in energy and the other in track length. Secondly, we compare the existing dose computation method in _Chronholm and Pryer_ with a smoother alternative using finite element methods, which was developed in the adjoining thesis to this work. Thirdly, we compare the use of these methods with standard Monte Carlo against Multilevel Monte Carlo for this task, alongside suggesting techniques for ensuring dose positivity.

Overall, the main contribution of this work was that, using the new simulation and dose computation method we developed, the MLMC was successful at achieving significantly improved computational complexity over standard Monte Carlo, in the homogeneous medium tested. Future work would aim to extend this work to include other media, inhomogeneity, more complex geometric structures, and explore other variants of MLMC such as antithetic MLMC. Further details are given in the thesis.

## Installation

### HPC Hardware

Due to the high objective computational complexity of this task, access to a High-Performance Computing (HPC) cluster is recommended for running large-scale experiments. For example, running MLMC with root mean-square-error $\epsilon < 10^{-1}$ is likely to require at least $\mathcal{O}(10^{7})$ proton simulations in 2D, based on our use of this code. In 3D, we expect the complexity to worsen significantly for all methods with the current setup.

The experiments presented in this thesis were performed using the an HPC cluster owned by the University of Oxford. 128 CPU cores were used over which proton simulations were run in parallel. Linux OS and no job scheduler was used. One full dose map under the conditions described above in 2D was produced in the order of hours. Note that improving the error requirement can notably increase the runtime. Further information about the HPC cluster can be provided upon reasonable request.

Smaller plots (e.g. large $\epsilon \sim \\mathcal{O}(1)$, Monte Carlo dose maps using $\mathcal{O}(10^{5})$ simulations) and unit tests may be executed locally.

### Software Requirements

The following software is required:

- Python [minimum: 3.11]
- All standard scientific computing libraries, plus Numba [version requirements not yet determined]

### Installation from Source

The project is intended to be distributed as an installable Python package, with its dependencies managed through [`pyproject.toml`](pyproject.toml). Currently, the project can be installed directly from the repository. Below we give installation instructions for Mac users.

Clone the repository:

```bash
git clone https://github.com/kezoltan/MSc-Thesis-Proton-Therapy-Monte-Carlo-.git
cd MSc-Thesis-Proton-Therapy-Monte-Carlo-
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
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

Create a user-owned TOML config file from the packaged example:

```bash
pbt-mlmc --write-config dose-config.toml
```

Edit `dose-config.toml` to set the simulation method, dose parameters, beam and mesh properties, MLMC settings, and output directory. The generated example selects a small 2D standard Monte Carlo run (`NUM_CPUS = 1`, `SIMS_PER_CPU = 10`). Start it with:

```bash
pbt-mlmc --config dose-config.toml
```

The config uses TOML sections: `[output]`, `[simulation]`, `[beam]`, `[mesh]`, and `[mlmc]`. Parameter names match the existing Python settings, including `E0`, `side_len`, `SPATIAL_DIM`, and `Eps`. Unknown names and invalid values are reported before simulation modules are loaded. Values omitted from a custom config use the package defaults.

Paths in `file_path` are resolved from the current working directory. The selected output directory is created if needed. Results and the generated lumped mass-matrix cache are stored there. No external input data files are required.

The package defaults are documented in `pbt_mlmc/simulation_config.py`; routine user changes should go in the external TOML file so they survive package upgrades. The primary user choices and defaults are `file_path` (`/tmp/CHANGE_THIS`), `SPATIAL_DIM` (`2`), `sde_model` (`energy`), `dose_method` (`SF`), `sampling_type` (`mc`), `SIMS_PER_CPU` (`200`), and `Eps` (`[0.2, 0.1]`). Valid model choices are `energy` or `track_length`; dose methods are `SF` or `SK`; sampling methods are `mc` or `mlmc`. `Eps` is used by MLMC and `SIMS_PER_CPU` is primarily used for standard Monte Carlo. The shipped TOML is a small example configuration and may intentionally use smaller run values than these defaults.

The command should print simulation details before it starts, for example:

```bash
"Gaussian energy spread is inactive: energy deviation is 0, E0 is 62."
"Gaussian initial position sampling is active."
"Simulation details: mc dose method SF, model energy, 0.03 h, EO 62, side_len 0.05, N 10, KAPPA 4e-05"
"Mass matrix loaded successfully."
```

Each run creates files in the configured output directory. Review or remove those results when they are no longer needed.

### Future Package Distribution

A console command, `pbt-mlmc`, is installed with the package. 

*Note: PyPI installation is not yet available - [Issue #13](../../issues/13). We will update this section in due coarse.*
### Parameterisation

We provide default parameters for most variables for the purposes of testing, however we expect to provide alternative presets (e.g. for different materials) in future. Currently, the user should pay attention to:

| Config setting | Function | Suggested Default* |
|----------|----------|----------|
| sde_model | Select which proton SDE is used | 'energy' |
| dose_method | Select which dose computation technique is used | 'SF' |
| sampling_type | Select which Monte Carlo method is used | 'mc' |
| file_path | Select where data and pictures are saved | `./pbt_mlmc_output` in the example config |
| SPATIAL_DIM | Select the number of spatial dimensions in which to estimate | 2 |
| Eps | List of RMSE values, MLMC only | $\geq 0.2$ |
| SIMS_PER_CPU | Number of simulations per CPU core, Monte Carlo only | _hardware dependent_ |

*The suggested defaults above are based on the results of experiments but are not restrictive. The first three parameter settings produced the best overall computational complexity (when combined) and hence are recommended, but these results have yet to be reproduced. If reproducing the thesis results, other values of the first three settings will be required for comparison and no other parameters (excluding the file_path) should be modified.

### Example Output

Depending on the user's choice of Monte Carlo method, different figures will be generated and saved to the user's machine in their elected folder. Additionally, different parameterisations will create spatial dose maps with different characteristics - for example, higher energy beams will have a higher range. Below we provide one example of a dose map created for a beam generated using all existing default parameter settings:

![example monte carlo dose map](https://github.com/kezoltan/MSc-Thesis-Proton-Therapy-Monte-Carlo-/blob/main/docs/figures/mc_SF_energy.png)

*Example output of proton beam dose map generated using the standard Monte Carlo sampling method with the energy parameterised SDE model and the (so-called) 'shape function' dose estimation technique in 2D. The mean initial proton energy is 62 MeV and the straggling strength is 'Moderate' with* $\kappa = 4 \times 10^{-5}$.

## Testing and Validation

Reproducible regimes for testing are currently being implemented [Issue #7](../../issues/7). As such, **we do not recommend installing this work for any reason other than reproducing results at this time.**

## Use of Generative AI

ChatGPT (OpenAI) was used to assist with selected code-writing, refactoring, and debugging tasks. The project authors are responsible for the final code, scientific content, and results.


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

See ['CONTRIBUTING.md'](https://github.com/kezoltan/MSc-Thesis-Proton-Therapy-Monte-Carlo-/blob/main/CONTRIBUTING.md) for further information.

## License

This project is distributed under the MIT License. See ['LICENSE.md'](https://github.com/kezoltan/MSc-Thesis-Proton-Therapy-Monte-Carlo-/blob/main/LICENSE.md) for details.

## Contact

For questions regarding this project, please contact:

Katie Zoltan (https://github.com/kezoltan)
katherine.zoltan@bnc.ox.ac.uk
