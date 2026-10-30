Dose parameters
===============

User settings are provided through a TOML file passed to the CLI with
``pbt-mlmc --config path/to/config.toml``. The CLI validates the file before
loading the simulation. Settings omitted from a custom file use the defaults
listed in ``pbt_mlmc.simulation_config`` and ``example_config.toml``.

The primary user choices are:

* ``file_path`` (``/tmp/CHANGE_THIS``): output directory; relative paths are
  resolved from the current working directory.
* ``SPATIAL_DIM`` (``2``): spatial dimension, either 2 or 3.
* ``sde_model`` (``energy``): ``energy`` or ``track_length``.
* ``dose_method`` (``SF``): ``SF`` (shape function) or ``SK`` (spatial kernel).
* ``sampling_type`` (``mc``): ``mc`` or ``mlmc``.
* ``SIMS_PER_CPU`` (``200``): simulations per CPU, primarily for standard
  Monte Carlo.
* ``Eps`` (``[0.2, 0.1]``): positive error tolerances used by MLMC.

The TOML file also supports beam, mesh, CPU, seed, and MLMC tuning options;
``example_config.toml`` lists these settings and their example values.
The CPU count is derived from the available CPUs. Energy and initial-position
spread activation are derived from ``energy_sdev`` and ``width_sdev_factor``;
these derived values should not be set in the TOML file.

.. automodule:: pbt_mlmc.simulation_config
   :members:
   :undoc-members:

