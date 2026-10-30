"""Read and validate user-owned TOML simulation settings."""

import math
import os
import tomllib
from pathlib import Path

CONFIG_FILE_ENV = "PBT_MLMC_CONFIG"

SECTION_KEYS = {
    "output": {"file_path"},
    "simulation": {
        "SPATIAL_DIM", "sde_model", "dose_method", "sampling_type",
        "SIMS_PER_CPU", "NUM_CPUS", "Eps", "random_seed",
    },
    "beam": {
        "E0", "EMIN", "ALPHA", "p", "RHO", "EPS_0", "straggling_severity",
        "width_sdev_factor", "width_spread", "energy_spread_percent", "energy_spread", "energy_sdev",
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
    "SPATIAL_DIM", "SIMS_PER_CPU", "NUM_CPUS", "random_seed", "MLMC_LEVEL_OFFSET",
    "L_conv_test", "N_conv_test", "Lmax", "N0", "base", "Lmin",
}
BOOLEAN_KEYS = {"width_spread", "energy_spread"}
NUMBER_KEYS = {
    "E0", "EMIN", "ALPHA", "p", "RHO", "EPS_0", "width_sdev_factor",
    "energy_spread_percent", "energy_sdev", "side_len", "range_allowance", "y_scaling",
}
POSITIVE_KEYS = {
    "SIMS_PER_CPU", "NUM_CPUS", "E0", "EMIN", "ALPHA", "p", "RHO", "side_len",
    "range_allowance", "y_scaling", "N_conv_test", "Lmax", "N0", "base", "L_conv_test",
}


def read_config(path: str | os.PathLike[str]) -> dict:
    """Read a TOML file, flatten its sections, and reject invalid settings."""
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
    for section, settings in document.items():
        if not isinstance(settings, dict):
            raise ValueError(f"[{section}] must contain key/value settings.")
        unknown_keys = settings.keys() - SECTION_KEYS[section]
        if unknown_keys:
            raise ValueError(f"Unknown setting(s) in [{section}]: {', '.join(sorted(unknown_keys))}")
        values.update(settings)

    _validate(values)
    if "file_path" in values:
        output_path = Path(values["file_path"]).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        values["file_path"] = str(output_path.resolve())
    return values


def _validate(values: dict) -> None:
    for key, allowed in CHOICES.items():
        if key in values and (not isinstance(values[key], str) or values[key] not in allowed):
            raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}.")

    for key in INTEGER_KEYS & values.keys():
        if type(values[key]) is not int:
            raise ValueError(f"{key} must be an integer.")
    for key in BOOLEAN_KEYS & values.keys():
        if type(values[key]) is not bool:
            raise ValueError(f"{key} must be true or false.")
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
