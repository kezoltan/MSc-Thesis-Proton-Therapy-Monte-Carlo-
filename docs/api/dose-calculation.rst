Dose Computation in Space
================

Given one proton path simulation, these functions use one of two 
methods to compute the dose deposit of that proton onto the 
nodes in the domain defined in src/parameterisation and 
src/setup_functions. The methods are referred to as either the 
'spatial kernel' or 'shape function' method depending on which 
numerical technique is used to smooth dose throughout space. 
By convention, the spatial kernel is a Gaussian function and the
shape functions are bilinear/trilinear.

Spatial Kernel Method
---------------------

.. automodule:: dose_computation.dosemap2_spatial_kernel_geoEM
   :members:

Shape Function Method
---------------------

.. automodule:: dose_computation.dosemap1_shape_function_geoEM
   :members:

