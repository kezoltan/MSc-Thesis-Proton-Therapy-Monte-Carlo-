#### Parameterisation Guidelines

###### Units

Lengths: cm
Mass: g
Energy: MeV

###### General Information

Parameters must be assigned for each of the following purposes:

- Choose the numerical experiment: the type of Monte Carlo experiment, SDE model and dose computation method used
- Choose a location to save output data 
- Set any experimental properties e.g. number of simulations, accuracy, MLMC levels
- Set physical properties e.g. medium dependent parameters, beam settings
- Based on the above, determine conditional parameters

For installed-package use, create a user-owned TOML file with `pbt-mlmc --write-config dose-config.toml` and edit the settings there. The config file is read before dependent mesh and beam values are calculated. For development, `core_params.py` and `_fixed_params.py` still provide the package defaults and can be changed in an editable checkout.

Do not modify parameters given in `_conditional_params.py` or its setup functions; those values are derived from the user settings.

Please read the comments in the former two files for any known parameter restrictions.
