"""
tab_merger.py
Two responsibilities:
1. Parse the compound_blocks layout in the Data Analysis tab into a clean DataFrame.
2. Merge all extracted tabs into a single working DataFrame keyed on compound_id.

The Data Analysis block structure (Pharmaron kinetic solubility):
  Block header row : col A = compound index (int or "PC"), col B = compound name
  IS row           : col A = None, col B = IS name, col C = sample_id (contains "Blank")
  Replicate rows   : col A = None, col B = None, col C = sample_id (contains "Sample")
  Average row      : col A = None, col B = None, col C = sample_id (ends with "-2" by convention)
  STD row          : col A = None, col B = None, col C = sample_id (contains "STD")
  Blank row        : col A = None, col B = None (row after header, first data row)

Because the number of replicates can vary, we detect rows dynamically by
inspecting the sample_id string content.
"""

from __future__ import annotations
import re
import pandas as pd
from typing import Any


# ── Public entry point ────────────────────────────────────────────────────────

def parse_data_analysis(raw_rows: list[tuple], col_config: list[dict]) -> pd.DataFrame:
    """
    Parse the compound_blocks Data Analysis tab into a structured DataFrame.
    Each output row represents one measurement row (blank, replicate, average, or STD).
    A 'compound_id' and 'row_type' column are added.

    row_type values: 'blank', 'replicate', 'average', 'std'
    """
    records = []
    current_compound: str | None = None
    current_is: str | None = None
    current_index: Any = None

    i = 0
    while i < len(raw_rows):
        row = raw_rows[i]
        # Detect block header: col A is non-None (int or "PC") AND col B has a name
        if _is_block_header(row):
            current_index = row[0]
            current_compound = str(row[1]).strip() if row[1] is not None else None
            current_is = None
            i += 1
            continue

        # IS / blank row: col A is None, col B has IS name, col C has sample_id with "Blank"
        if (current_compound is not None
                and row[0] is None
                and row[1] is not None
                and row[2] is not None
                and "blank" in str(row[2]).lower()):
            current_is = str(row[1]).strip()
            rec = _build_record(row, col_config, current_compound, current_is,
                                current_index, "blank")
            records.append(rec)
            i += 1
            continue

        # Data rows: col A None, col B None or IS-name, col C has sample_id
        if current_compound is not None and row[0] is None and row[2] is not None:
            sample_id = str(row[2]).strip()
            row_type = _classify_row(sample_id)
            if row_type is not None:
                rec = _build_record(row, col_config, current_compound, current_is,
                                    current_index, row_type)
                records.append(rec)

        i += 1

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Relabel the last 'replicate' row per compound as 'average'
    # (CRO convention: last numbered row stores AVERAGE of the preceding replicates)
    for compound in df["compound_id"].unique():
        mask = (df["compound_id"] == compound) & (df["row_type"] == "replicate")
        indices = df.index[mask].tolist()
        if len(indices) >= 2:
            df.loc[indices[-1], "row_type"] = "average"

    df = pd.DataFrame(records)

    # Apply dtypes for numeric columns
    numeric_cols = [c["canonical"] for c in col_config if c.get("dtype") == "float"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def merge_tabs(
    tabs: dict[str, Any],
    column_map: dict,
) -> dict[str, Any]:
    """
    Merge tabular tabs onto the Summary DataFrame using compound_id as key.
    Returns the tabs dict with an added "merged" key containing the full
    working DataFrame for test compounds.

    Non-tabular tabs (signature dict, study_design_text str, data_analysis df)
    are passed through unchanged.
    """
    col_config = column_map["tabs"]["data_analysis"]["columns"]
    tabs["data_analysis_parsed"] = parse_data_analysis(
        tabs["data_analysis"], col_config
    )

    summary_df: pd.DataFrame = tabs["summary"]
    materials_df: pd.DataFrame = tabs["materials"]
    bioanalytical_df: pd.DataFrame = tabs["bioanalytical"]

    # Merge materials onto summary (MW, FW)
    merged = summary_df.merge(
        materials_df[["compound_id", "mw_gmol", "fw_gmol"]],
        on="compound_id",
        how="left",
    )

    # Merge bioanalytical onto summary (Q1, DP, CE)
    merged = merged.merge(
        bioanalytical_df[["compound_id", "q1_mz", "dp_v", "ce_v"]],
        on="compound_id",
        how="left",
    )

    tabs["merged"] = merged
    return tabs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_block_header(row: tuple) -> bool:
    """
    A block header row has a non-None value in col A (int or the string "PC")
    and a non-None compound name in col B.
    """
    col_a = row[0] if len(row) > 0 else None
    col_b = row[1] if len(row) > 1 else None
    if col_a is None or col_b is None:
        return False
    col_a_str = str(col_a).strip()
    return col_a_str != "" and col_b != ""


def _classify_row(sample_id: str) -> str | None:
    """
    Classify a data row by its sample_id string.
    Returns: 'replicate_label', 'replicate', 'average', 'std', or None (skip).

    Pharmaron CRO block structure:
      Sample-1000        → group label row (ratio not used in CRO formula, skip for calcs)
      Sample-1000-1      → true replicate 1
      Sample-1000-2      → average row (CRO stores AVERAGE(H4:H5) here)
      STD                → standard row
    The distinguishing rule: if the sample_id ends in a replicate suffix
    matching '-<single_digit>' the entry is a numbered replicate or average;
    the unsuffixed 'Sample-NNN' row is the group label.
    We mark the last numbered row as 'average' and all preceding as 'replicate'.
    Because we don't know the total count at classify time, we mark all numbered
    rows as 'replicate' here and let parse_data_analysis relabel the last one.
    """
    sid = sample_id.lower()
    if "std" in sid:
        return "std"
    if "sample" in sid:
        # Check for trailing replicate index: ends with -<digits> after the dilution level
        # e.g. Sample-1000-1, Sample-100-2 → numbered replicates/average
        # vs   Sample-1000, Sample-100     → group label row (not used in CRO formula)
        if re.search(r"-\d+$", sample_id):
            # Has a trailing numeric suffix after the dilution number
            parts = sample_id.rsplit("-", 1)
            # Make sure the last part is a single replicate index, not the dilution level
            last = parts[-1]
            second_last = parts[0].rsplit("-", 1)[-1] if "-" in parts[0] else ""
            # Dilution levels are typically 100/1000/10000; replicate indices are 1,2,3
            try:
                if int(last) < 10:  # replicate index (1, 2, 3...)
                    return "replicate"    # will relabel last one to 'average' after block
            except ValueError:
                pass
        # Unsuffixed group label row — skip for calculations
        return "replicate_label"
    return None


def _build_record(
    row: tuple,
    col_config: list[dict],
    compound_id: str,
    is_name: str | None,
    compound_index: Any,
    row_type: str,
) -> dict:
    rec: dict = {
        "compound_id": compound_id,
        "is_name": is_name,
        "compound_index": compound_index,
        "row_type": row_type,
    }
    for col_cfg in col_config:
        idx = col_cfg["col_index"]
        canonical = col_cfg["canonical"]
        val = row[idx] if idx < len(row) else None
        rec[canonical] = val
    return rec
