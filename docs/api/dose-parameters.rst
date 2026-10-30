Dose parameters
===============

Create a user-owned TOML file with ``pbt-mlmc --write-config dose-config.toml``
and run it with ``pbt-mlmc --config dose-config.toml``. The bundled
``example_config.toml`` is deliberately reduced to ``[output]`` and
``[simulation]``. The CLI validates the TOML before loading the simulation;
omitted settings use built-in defaults.

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

The reduced example contains only those seven user choices. Optional tuning
settings are also accepted: ``random_seed`` in ``[simulation]``; beam values
``E0``, ``EMIN``, ``ALPHA``, ``p``, ``RHO``, ``EPS_0``,
``straggling_severity``, ``width_sdev_factor``, ``energy_spread_percent``, and
``energy_sdev`` in ``[beam]``; ``side_len``, ``range_allowance``, and
``y_scaling`` in ``[mesh]``; and ``MLMC_LEVEL_OFFSET``, ``L_conv_test``,
``N_conv_test``, ``Lmax``, ``N0``, ``base``, and ``Lmin`` in ``[mlmc]``.

``NUM_CPUS`` is calculated from available CPUs. ``width_spread`` and
``energy_spread`` are calculated from ``width_sdev_factor`` and
``energy_sdev``. They are derived values and are not valid TOML options. If a
config overrides any of ``ALPHA``, ``p``, or ``RHO``, it must provide all three
because they describe the same physical model.

.. automodule:: pbt_mlmc.simulation_config
   :members:
   :undoc-members:

