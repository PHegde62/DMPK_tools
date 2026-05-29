"""
standardizer.py
Applies std_rules.yaml transformations to the merged DataFrame.
Produces a clean upload-ready DataFrame with canonical fields standardised.
"""

from __future__ import annotations
import re
import pandas as pd
from typing import Any


def apply_std_rules(
    merged_df: pd.DataFrame,
    signature: dict[str, Any],
    std_rules: dict[str, Any],
    da_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Apply all transformations from std_rules.yaml.
    Returns a new DataFrame ready for upload_writer.

    Parameters
    ----------
    merged_df  : summary + materials + bioanalytical merged DataFrame
    signature  : key_value dict from Signature tab
    std_rules  : the "std_rules" config dict
    da_df      : parsed Data Analysis DataFrame (for BLQ detection)
    """
    df = merged_df.copy()

    # Attach CRO name from Signature tab
    raw_cro = signature.get("cro_name", "")
    df["cro_name"] = _extract_cro_name(str(raw_cro))

    # Apply comment rules (BLQ/BLD detection)
    df["comment"] = ""
    blq_cfg = _get_blq_config(std_rules)
    if blq_cfg and da_df is not None:
        blq_compounds = _detect_blq(da_df, blq_cfg)
        mask = df["compound_id"].isin(blq_compounds)
        df.loc[mask, "comment"] = blq_cfg.get("comment_text", "below lower limit of detection")

    # Apply field transformations
    for transform in std_rules.get("transformations", []):
        df = _apply_transform(df, transform)

    return df


def apply_test_compound_filter(
    df: pd.DataFrame,
    filter_cfg: dict,
) -> pd.DataFrame:
    """
    Return only rows matching the test compound filter.
    strategy: "prefix" → keep rows where compound_id starts with the prefix.
    """
    strategy = filter_cfg.get("strategy", "prefix")
    if strategy == "prefix":
        prefix = filter_cfg.get("prefix", "GEN-")
        mask = df["compound_id"].str.strip().str.upper().str.startswith(prefix.upper())
        return df[mask].reset_index(drop=True)
    raise ValueError(f"Unknown filter strategy: {strategy}")


# ── Transform implementations ─────────────────────────────────────────────────

def _apply_transform(df: pd.DataFrame, transform: dict) -> pd.DataFrame:
    field = transform["field"]
    operation = transform["operation"]
    out_field = transform.get("output_field", field)

    if field not in df.columns:
        return df

    if operation == "round":
        dp = int(transform.get("decimal_places", 2))
        df[out_field] = pd.to_numeric(df[field], errors="coerce").round(dp)

    elif operation == "strip_whitespace":
        df[out_field] = df[field].astype(str).str.strip()

    elif operation == "extract_cro_name":
        df[out_field] = df[field].apply(lambda v: _extract_cro_name(str(v)))

    elif operation == "multiply":
        factor = float(transform["factor"])
        df[out_field] = pd.to_numeric(df[field], errors="coerce") * factor

    elif operation == "map":
        mapping = transform.get("mapping", {})
        df[out_field] = df[field].map(mapping).fillna(df[field])

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_cro_name(raw: str) -> str:
    """
    Extract a clean CRO name from a freeform 'Submitted by' string.
    E.g. "in vitro ADME laboratory, Pharmaron" → "Pharmaron"
    Strategy: take the last comma-delimited token and strip it.
    If no comma, return the full string stripped.
    """
    if not raw or raw == "None":
        return ""
    parts = [p.strip() for p in raw.split(",")]
    return parts[-1] if parts else raw.strip()


def _get_blq_config(std_rules: dict) -> dict | None:
    """Pull BLQ detection config from qc_rules — passed via std_rules for convenience."""
    return std_rules.get("blq_detection")


def _detect_blq(da_df: pd.DataFrame, blq_cfg: dict) -> set[str]:
    """Return compound_ids that have BLQ/BLD text in any search field."""
    keywords = [k.lower() for k in blq_cfg.get("keywords", ["BLQ", "BLD"])]
    search_fields = blq_cfg.get("search_fields", ["sample_id"])
    blq_compounds: set[str] = set()

    for field in search_fields:
        if field not in da_df.columns:
            continue
        mask = da_df[field].astype(str).str.lower().apply(
            lambda v: any(kw in v for kw in keywords)
        )
        blq_compounds.update(da_df.loc[mask, "compound_id"].tolist())

    return blq_compounds
