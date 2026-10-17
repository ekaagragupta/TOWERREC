"""Configuration loading, merging, and validation using OmegaConf / PyYAML."""

from pathlib import Path
from typing import Any

import yaml

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    OmegaConf = None
    DictConfig = None


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> Any:
    """Loads a YAML configuration file with optional overrides.

    Args:
        config_path: Path to YAML config.
        overrides: Dictionary of dot-separated keys to override.

    Returns:
        Config object (DictConfig if OmegaConf is available, else native dict).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    if OmegaConf is not None:
        cfg = OmegaConf.load(str(path))
        if overrides:
            override_cfg = OmegaConf.create(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)
        return cfg
    else:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if overrides:
            for k, v in overrides.items():
                parts = k.split(".")
                d = cfg
                for p in parts[:-1]:
                    d = d.setdefault(p, {})
                d[parts[-1]] = v
        return cfg


def to_dict(config: Any) -> dict[str, Any]:
    """Converts a DictConfig or dict to a standard Python dictionary."""
    if OmegaConf is not None and isinstance(config, (DictConfig,)):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore
    elif isinstance(config, dict):
        return config
    return dict(config)
