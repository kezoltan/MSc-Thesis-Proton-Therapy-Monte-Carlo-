"""Command-line entry point for a dose simulation."""

import argparse
import os
from importlib.resources import files
from pathlib import Path

from pbt_mlmc.config import CONFIG_FILE_ENV, read_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a proton dose simulation.")
    parser.add_argument("--config", type=Path, help="TOML file with simulation settings")
    parser.add_argument("--write-config", type=Path, metavar="PATH", help="write an example config and exit")
    args = parser.parse_args()

    if args.config and args.write_config:
        parser.error("choose either --config or --write-config")

    if args.write_config:
        path = args.write_config.expanduser().resolve()
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        example = files("pbt_mlmc").joinpath("default_config.toml").read_text(encoding="utf-8")
        path.write_text(example, encoding="utf-8")
        print(f"Wrote example config to {path}")
        return

    if args.config:
        path = args.config.expanduser().resolve()
        try:
            settings = read_config(path)
            output_dir = Path(settings.get("file_path", "/tmp/CHANGE_THIS"))
            output_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if not os.access(output_dir, os.W_OK):
            parser.error(f"output directory is not writable: {output_dir}")
        os.environ[CONFIG_FILE_ENV] = str(path)

    # Parameter modules calculate derived values on import, so load them after the config.
    from pbt_mlmc.dose_main import main as run_simulation

    run_simulation()


if __name__ == "__main__":
    main()
