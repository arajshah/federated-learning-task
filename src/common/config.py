from pathlib import Path
from typing import Any

import yaml


Config = dict[str, Any]


def load_config(config_path: str | Path) -> Config:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return config
