"""
config_loader.py
Loads all YAML config files for a given assay type.
Returns typed dicts consumed by every other module.
No assay-specific logic lives here — it's a pure file reader.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any


CONFIG_ROOT = Path(__file__).parent.parent / "config" / "assay_config"


def load_assay_config(assay_type: str) -> dict[str, Any]:
    """
    Load column_map, qc_rules, and std_rules for the given assay_type.

    Parameters
    ----------
    assay_type : str
        Folder name under config/assay_config/, e.g. "kinetic_solubility"

    Returns
    -------
    dict with keys: "column_map", "qc_rules", "std_rules"
    """
    assay_dir = CONFIG_ROOT / assay_type
    if not assay_dir.is_dir():
        available = [d.name for d in CONFIG_ROOT.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"No config found for assay type '{assay_type}'. "
            f"Available: {available}"
        )

    config: dict[str, Any] = {}
    for config_file in ("column_map", "qc_rules", "std_rules"):
        path = assay_dir / f"{config_file}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Missing config file: {path}")
        with path.open("r", encoding="utf-8") as fh:
            config[config_file] = yaml.safe_load(fh)

    return config


def get_upload_template_path(assay_type: str) -> Path:
    """Return path to the upload template Excel for this assay type."""
    path = CONFIG_ROOT / assay_type / "upload_template.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Upload template not found: {path}")
    return path


def get_control_compounds(config: dict) -> list[str]:
    return config["column_map"].get("control_compounds", [])


def get_test_compound_filter(config: dict) -> dict:
    return config["column_map"].get("test_compound_filter", {"strategy": "prefix", "prefix": "GEN-"})
