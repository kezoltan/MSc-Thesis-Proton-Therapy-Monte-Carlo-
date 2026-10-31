Monte Carlo Methods and Parallelisation
===================

These functions are required to execute standard Monte Carlo 
and (geometric) MLMC in parallel. Furthermore, this section 
includes MLMC driver code written in Python in the usual style
of Mike Giles, see `this link`_. 
.. _this link: https://people.maths.ox.ac.uk/~gilesm/mlmc/ 
alongside modifications and additional data collection that 
were required for this project.

MLMC Driver Code

---------------------

.. automodule:: monte_carlo_methods.mlmc_driver.mlmc_main
   :members:

.. automodule:: monte_carlo_methods.mlmc_driver.mlmc_test
   :members:

.. automodule:: monte_carlo_methods.mlmc_driver.mlmc_plot
   :members:

Worker Functions and Parallelisation

---------------------

.. automodule:: monte_carlo_methods.mc_parallel
   :members:

.. automodule:: monte_carlo_methods.mlmc_parallel
   :members:

