from __future__ import annotations

from pathlib import Path
from typing import Optional
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "fpr_config.yaml"


def resolve_rel_path(value: Optional[str], base_dir: Path) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def load_config(config_path: Optional[str | Path] = None) -> dict:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_dir = cfg_path.parent
    paths = cfg.get("paths", {})
    resolved_paths = {}
    for key, val in paths.items():
        if isinstance(val, str) and key != "subjects_glob":
            resolved_paths[key] = resolve_rel_path(val, base_dir)
        else:
            resolved_paths[key] = val
    cfg["paths"] = resolved_paths
    cfg["_config_path"] = str(cfg_path)
    cfg["_base_dir"] = str(base_dir)
    return cfg


def script_path(script_name: str) -> Path:
    return PACKAGE_ROOT / f"{script_name}.py"
