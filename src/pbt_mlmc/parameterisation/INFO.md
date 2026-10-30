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

We refer all users to choose the parameters in [core_params.py]('core_params.py'), where the numerical experiment is defined. It is likely that modifications to [_fixed_params.py]('_fixed_params.py') will also be needed, but alter these parameters with slightly more caution - the main reason to touch these would be (1) to alter the simulation to align with your hardware capability and (2) to investigate different physical parameterisations. 

We recommend not modifying parameters given in [_conditional_params.py]('_conditional_params.py') or any of their setup functions as these are assigned based on conventions. 

Please read the comments in the former two files for any known parameter restrictions.