"""Simulation settings loaded from the active TOML configuration."""

import os

import numpy as np

from pathlib import Path
import math
import tomllib

CONFIG_FILE_ENV = "PBT_MLMC_CONFIG"

# splitting all the variables into type lists for easier validation
SECTION_KEYS = {
    "output": {"file_path"},
    "simulation": {
        "SPATIAL_DIM", "sde_model", "dose_method", "sampling_type",
        "SIMS_PER_CPU", "Eps", "random_seed",
    },
    "beam": {
        "E0", "EMIN", "ALPHA", "p", "RHO", "EPS_0", "straggling_severity",
        "width_sdev_factor", "energy_spread_percent", "energy_sdev",
    },
    "mesh": {"side_len", "range_allowance", "y_scaling"},
    "mlmc": {"MLMC_LEVEL_OFFSET", "L_conv_test", "N_conv_test", "Lmax", "N0", "base", "Lmin"},
}

CHOICES = {
    "sde_model": {"energy", "track_length"},
    "dose_method": {"SF", "SK"},
    "sampling_type": {"mc", "mlmc"},
    "straggling_severity": {"None", "Light", "Moderate", "Strong"},
}
INTEGER_KEYS = {
    "SPATIAL_DIM", "SIMS_PER_CPU", "random_seed", "MLMC_LEVEL_OFFSET",
    "L_conv_test", "N_conv_test", "Lmax", "N0", "base", "Lmin",
}
NUMBER_KEYS = {
    "E0", "EMIN", "ALPHA", "p", "RHO", "EPS_0", "width_sdev_factor",
    "energy_spread_percent", "energy_sdev", "side_len", "range_allowance", "y_scaling",
}
POSITIVE_KEYS = {
    "SIMS_PER_CPU", "E0", "EMIN", "ALPHA", "p", "RHO", "side_len",
    "range_allowance", "y_scaling", "N_conv_test", "Lmax", "N0", "base", "L_conv_test",
}

# Alpha, p, and RHO describe the same physical model and should be changed together.




def read_config(path: str | os.PathLike[str]) -> dict:
    """Read and validate a TOML file, returning flattened settings."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as file:
            document = tomllib.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Config file does not exist: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {config_path}: {exc}") from exc

    unknown_sections = document.keys() - SECTION_KEYS.keys()
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {', '.join(sorted(unknown_sections))}")

    values = {}
    for section, section_settings in document.items():
        if not isinstance(section_settings, dict):
            raise ValueError(f"[{section}] must contain key/value settings.")
        unknown_keys = section_settings.keys() - SECTION_KEYS[section]
        if unknown_keys:
            raise ValueError(f"Unknown setting(s) in [{section}]: {', '.join(sorted(unknown_keys))}")
        values.update(section_settings)

    _validate(values)
    if "file_path" in values:
        output_path = Path(values["file_path"]).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        values["file_path"] = str(output_path.resolve())
    return values


def _validate(values: dict) -> None:
    medium_model_keys = {"ALPHA", "p", "RHO"}
    if medium_model_keys.intersection(values) and not medium_model_keys.issubset(values):
        missing_keys = ", ".join(sorted(medium_model_keys - values.keys()))
        raise ValueError(
            "ALPHA, p, and RHO describe the same physical model and must be set together; "
            f"missing: {missing_keys}."
        )

    for key, allowed in CHOICES.items():
        if key in values and (not isinstance(values[key], str) or values[key] not in allowed):
            raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}.")
    for key in INTEGER_KEYS & values.keys():
        if type(values[key]) is not int:
            raise ValueError(f"{key} must be an integer.")
    for key in NUMBER_KEYS & values.keys():
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
    if "file_path" in values and not isinstance(values["file_path"], str):
        raise ValueError("file_path must be a string path.")
    if "Eps" in values:
        eps = values["Eps"]
        if not isinstance(eps, list) or not eps or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in eps
        ):
            raise ValueError("Eps must be a non-empty array of positive numbers.")
    for key in POSITIVE_KEYS & values.keys():
        if values[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    for key in {"width_sdev_factor", "energy_spread_percent", "energy_sdev", "EPS_0"} & values.keys():
        if values[key] < 0:
            raise ValueError(f"{key} must be zero or greater.")
    if values.get("SPATIAL_DIM", 2) not in {2, 3}:
        raise ValueError("SPATIAL_DIM must be 2 or 3.")
    if values.get("random_seed", 0) < 0:
        raise ValueError("random_seed must be zero or greater.")
    if values.get("EMIN", 0) >= values.get("E0", 62):
        raise ValueError("EMIN must be less than E0.")


_config_file = os.environ.get(CONFIG_FILE_ENV)
CONFIG = read_config(_config_file) if _config_file else {}

# User-configurable values: sourced from TOML, with defaults when omitted.
file_path = CONFIG.get("file_path", "/tmp/CHANGE_THIS")
SPATIAL_DIM = CONFIG.get("SPATIAL_DIM", 2)
sde_model = CONFIG.get("sde_model", "energy")
dose_method = CONFIG.get("dose_method", "SF")
sampling_type = CONFIG.get("sampling_type", "mc")
SIMS_PER_CPU = CONFIG.get("SIMS_PER_CPU", 200)
Eps = np.asarray(CONFIG.get("Eps", [0.2, 0.1]))

# Other configurable simulation inputs and their built-in defaults.
master_seed_seq = np.random.SeedSequence(CONFIG.get("random_seed", 42))
E0 = CONFIG.get("E0", 62)
ALPHA = CONFIG.get("ALPHA", 2.633e-3)
p = CONFIG.get("p", 1.735)
RHO = CONFIG.get("RHO", 1)
range_allowance = CONFIG.get("range_allowance", 1.6)
y_scaling = CONFIG.get("y_scaling", 1.0)
side_len = CONFIG.get("side_len", 0.05)
EMIN = CONFIG.get("EMIN", 0.1)
EPS_0 = CONFIG.get("EPS_0", 0.005)
straggling_severity = CONFIG.get("straggling_severity", "Moderate")
width_sdev_factor = CONFIG.get("width_sdev_factor", 3)
energy_spread_percent = CONFIG.get("energy_spread_percent", 1 / 100)
energy_sdev = CONFIG.get("energy_sdev", 0)
MLMC_LEVEL_OFFSET = CONFIG.get("MLMC_LEVEL_OFFSET", 5)
L_conv_test = CONFIG.get("L_conv_test", 8)
N_conv_test = CONFIG.get("N_conv_test", 200000)
Lmax = CONFIG.get("Lmax", 17)
N0 = CONFIG.get("N0", 50000)
base = CONFIG.get("base", 2)
Lmin = CONFIG.get("Lmin", 2)

# Derived runtime values: calculated from configuration or the current machine.
NUM_CPUS = max(1, (os.cpu_count() or 1) - 2)
sim_num = NUM_CPUS * SIMS_PER_CPU
origin = np.array([0.0, 0.0, 0.0])
len_reciprocal = 1 / side_len
T = E0 - EMIN
t1 = 1 / np.sqrt(3)
quad_nodes = [-t1, t1]
SIGMA = side_len * 3 / 4
width_spread = width_sdev_factor != 0 # boolean value depending on what width_sdev_factor is set to
energy_spread = energy_sdev != 0

# Mesh and model-dependent derived values.
from pbt_mlmc.setup_functions.nodal_mesh_setup import nodes, choose_X0  # noqa: E402
from pbt_mlmc.setup_functions.proton_setup_Bragg_Kleeman import choose_kappa, choose_Omega_0  # noqa: E402

KAPPA = choose_kappa()
Omega0 = choose_Omega_0()
if SPATIAL_DIM == 3:
    X_meshgrid, Y_meshgrid, Z_meshgrid = nodes(side_len)
    meshgrid = X_meshgrid, Y_meshgrid, Z_meshgrid
    nodes_array = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel(), Z_meshgrid.ravel()], axis=-1)
else:
    X_meshgrid, Y_meshgrid = nodes(side_len)
    meshgrid = X_meshgrid, Y_meshgrid
    nodes_array = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel()], axis=-1)
node_lookup = {tuple(np.rint(node / side_len).astype(int)): i for i, node in enumerate(nodes_array)}
X0_mean, domain_lower_bounds, domain_upper_bounds = choose_X0(meshgrid)
dose_shape = int(np.prod(X_meshgrid.shape))
max_steps = sum(X_meshgrid.shape)
title_seg = (
    f"{'Bilinear' if SPATIAL_DIM == 2 else 'Trilinear'} Basis Function"
    if dose_method == "SF"
    else "Spatial Kernel"
)
theta = 0.25 if dose_method == "SF" else 0.5
