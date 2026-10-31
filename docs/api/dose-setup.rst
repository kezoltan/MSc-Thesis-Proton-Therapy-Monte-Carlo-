Setup Functions and Conventions
=======================

These functions are required to setup all other functions of this code.
This includes proton simulation setup, some modelling assumptions used, 
and functions that recur throughout the other files.

Several important conventions are adhered in these files which could be 
modified for future research or other applications. For instance, the 
Bragg-Kleeman stopping power model is used in favour of other models such 
as the Bethe-Bloch model. We recommend reviewing all functions for any 
dependencies any existing conventions before making changes.

Unless otherwise stated, all functions should be assumed to be importing 
arguments from src/parameterisation. Given that these should not be 
locally modified, these parameters are not shown as keyword arguments.


Nodal Mesh Conventions and Mass Matrix Lumping
---------------------

.. automodule:: setup_functions.nodal_mesh_setup
   :members:

Mass Matrix Loading (continued) 
---------------------
.. automodule:: setup_functions.load_mass_matrix
   :members:

Gaussian Beam Sampling
---------------------

.. automodule:: setup_functions.gaussian_beam_setup
   :members:

Proton Transport Initialisation (Bragg-Kleeman) 
---------------------

.. automodule:: setup_functions.proton_setup_Bragg_Kleeman
   :members:
   
Shape Function Definition
---------------------

.. automodule:: setup_functions.shape_function_setup
   :members:

Geometric MLMC
---------------------

.. automodule:: setup_functions.mlmc_setup
   :members:

Parameter Verification
---------------------

.. automodule:: setup_functions.verify_params
   :members: