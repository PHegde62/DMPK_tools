"""
rule_evaluator.py
Stateless dispatcher for every QC check type defined in qc_rules.yaml.
To add a new check type: add check_<type>(tabs, rule, result) → QCResult,
then register it in REGISTRY. No other file changes needed.
"""

from __future__ import annotations
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QCResult:
    rule_id: str
    section: str
    description: str
    status: str        # "pass" | "fail" | "warn" | "manual"
    severity: str
    detail: str = ""
    affected_compounds: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def evaluate_rule(rule: dict[str, Any], tabs: dict[str, Any]) -> QCResult:
    check_type = rule.get("check")
    base = QCResult(
        rule_id=rule["id"],
        section=rule.get("section", ""),
        description=rule.get("description", ""),
        status="pass",
        severity=rule.get("severity", "warn"),
    )
    if check_type not in REGISTRY:
        base.status = "warn"
        base.detail = f"Unknown check type '{check_type}' — skipped"
        return base
    try:
        return REGISTRY[check_type](tabs, rule, base)
    except Exception as exc:
        base.status = "warn"
        base.detail = f"Evaluator error: {exc}"
        return base


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_true_replicates(grp: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with row_type == 'replicate' (not label, not average, not std)."""
    return grp[grp["row_type"] == "replicate"].copy()


def _get_dilution_from_blank(grp: pd.DataFrame) -> float | None:
    """
    The CRO stores the dilution factor as the evaluated IF() formula result
    in col G of the blank/IS row. Try to read it from there first.
    """
    blank_rows = grp[grp["row_type"] == "blank"]
    if blank_rows.empty:
        return None
    val = pd.to_numeric(blank_rows["dilution_or_std"].iloc[0], errors="coerce")
    if pd.notna(val) and val > 1:
        return float(val)
    return None


def _extract_dilution_from_sample_id(sample_id: str | None) -> float | None:
    if sample_id is None:
        return None
    m = re.search(r"-(\d{2,})", str(sample_id))
    if m:
        val = float(m.group(1))
        if val >= 10:
            return val
    return None


def _compute_solubilities(grp: pd.DataFrame) -> list[float]:
    """
    Compute individual solubility values (µM) for true replicates in a block.
    Formula: (sample_ratio / std_ratio) * std_conc * dilution_factor
    """
    std_rows = grp[grp["row_type"] == "std"]
    rep_rows = _get_true_replicates(grp)

    std_ratio = pd.to_numeric(std_rows["ratio"], errors="coerce").mean()
    std_conc  = pd.to_numeric(std_rows["dilution_or_std"], errors="coerce").mean()

    if pd.isna(std_ratio) or std_ratio == 0 or pd.isna(std_conc):
        return []

    dil = _get_dilution_from_blank(grp)

    sols = []
    for _, row in rep_rows.iterrows():
        ratio = pd.to_numeric(row["ratio"], errors="coerce")
        if pd.isna(ratio):
            continue
        d = dil if dil is not None else _extract_dilution_from_sample_id(row.get("sample_id"))
        if d is None:
            continue
        sols.append((ratio / std_ratio) * std_conc * d)
    return sols


# ── Check implementations ─────────────────────────────────────────────────────

def check_manual(tabs, rule, result):
    result.status = "manual"
    result.detail = rule.get("prompt", "Manual review required")
    return result


def check_compound_name_consistency(tabs, rule, result):
    """
    Check that compound IDs found in the primary 'summary' tab also appear
    in each of the other specified tabs (for the test compound only).
    Controls may legitimately be absent from the Materials tab.
    """
    tab_keys = rule.get("tabs", [])

    # Get the ground-truth set from Summary tab (only real compound IDs — short names)
    summary_df = tabs.get("summary")
    if not isinstance(summary_df, pd.DataFrame) or "compound_id" not in summary_df.columns:
        result.status = "warn"
        result.detail = "Summary tab not available for name consistency check"
        return result

    # Reference names: from summary, strip, upper, only plausible compound IDs (≤50 chars)
    ref_names = set(
        summary_df["compound_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    ref_names = {n for n in ref_names if len(n) <= 50}

    mismatches = []
    for key in tab_keys:
        if key == "summary":
            continue  # skip self-comparison
        tab = tabs.get(key)
        if tab is None and key == "data_analysis":
            tab = tabs.get("data_analysis_parsed")
        if not isinstance(tab, pd.DataFrame) or "compound_id" not in tab.columns:
            continue

        tab_names = set(
            tab["compound_id"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )
        tab_names = {n for n in tab_names if len(n) <= 50}

        # For Materials tab, controls may be absent (stored as narrative text, not table rows)
        # Only check that the TEST compound (GEN-prefixed) is present
        if key == "materials":
            test_only = {n for n in ref_names if n.startswith("GEN-")}
            absent = test_only - tab_names
        else:
            absent = ref_names - tab_names

        if absent:
            mismatches.append(f"{key}: missing {absent}")

    if mismatches:
        result.status = result.severity
        result.detail = "Name mismatch: " + "; ".join(mismatches)
    else:
        result.detail = "Compound names consistent across checked tabs"
    return result


def check_text_contains(tabs, rule, result):
    tab_key = rule.get("tab", "study_design_text")
    text = tabs.get(tab_key, "")
    if isinstance(text, dict):
        text = " ".join(str(v) for v in text.values() if v is not None)
    needle = rule.get("contains", "")
    if needle.lower() in str(text).lower():
        result.detail = f"'{needle}' confirmed present"
    else:
        result.status = result.severity
        result.detail = f"'{needle}' not found in {tab_key}"
    return result


def check_numeric_threshold(tabs, rule, result):
    tab_key    = rule["tab"]
    field_name = rule["field"]
    operator   = rule["operator"]
    threshold  = float(rule["threshold"])
    df = tabs.get(tab_key)
    if not isinstance(df, pd.DataFrame) or field_name not in df.columns:
        result.status = "warn"
        result.detail = f"Column '{field_name}' not found in '{tab_key}'"
        return result
    values = pd.to_numeric(df[field_name], errors="coerce")
    ops = {">=": lambda v: v >= threshold, "<=": lambda v: v <= threshold,
           ">":  lambda v: v > threshold,  "<":  lambda v: v < threshold,
           "==": lambda v: v == threshold}
    fn = ops.get(operator)
    if fn is None:
        result.status = "warn"; result.detail = f"Unknown operator '{operator}'"
        return result
    failing_mask = ~values.apply(fn) & values.notna()
    if failing_mask.any():
        result.status = result.severity
        failing_vals = values[failing_mask].tolist()
        cpds = df.loc[failing_mask, "compound_id"].tolist() if "compound_id" in df.columns else []
        result.affected_compounds = [str(c) for c in cpds]
        result.detail = (f"{len(failing_vals)} value(s) fail {field_name} {operator} {threshold}: "
                         + ", ".join(f"{v:.2f}" for v in failing_vals))
    else:
        result.detail = f"All {field_name} values satisfy {operator} {threshold}"
    return result


def check_q1_mz_match(tabs, rule, result):
    bio_df = tabs.get("bioanalytical")
    mat_df = tabs.get("materials")
    tolerance = float(rule.get("mz_tolerance_da", 0.5))
    if not isinstance(bio_df, pd.DataFrame) or not isinstance(mat_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Tabs not available"
        return result
    merged = bio_df.merge(mat_df[["compound_id", "mw_gmol"]], on="compound_id", how="inner")
    if merged.empty:
        result.status = "warn"; result.detail = "No matching compound IDs"
        return result
    failures = []
    for _, row in merged.iterrows():
        q1, mw = row.get("q1_mz"), row.get("mw_gmol")
        if pd.isna(q1) or pd.isna(mw): continue
        if abs(q1 - (mw + 1)) > tolerance:
            failures.append(f"{row['compound_id']}: Q1={q1:.3f}, expected≈{mw+1:.3f}")
    if failures:
        result.status = result.severity
        result.detail = "Q1 m/z mismatch: " + "; ".join(failures)
        result.affected_compounds = [f.split(":")[0] for f in failures]
    else:
        result.detail = f"All Q1 m/z within ±{tolerance} Da of MW+1"
    return result


def check_dilution_levels_present(tabs, rule, result):
    da_df = tabs.get("data_analysis_parsed")
    required = [str(lvl) for lvl in rule.get("required_levels", [])]
    if not isinstance(da_df, pd.DataFrame) or "sample_id" not in da_df.columns:
        result.status = "warn"; result.detail = "Data Analysis not available"
        return result
    combined = " ".join(da_df["sample_id"].dropna().astype(str).tolist())
    missing = [lvl for lvl in required if f"-{lvl}" not in combined]
    if missing:
        result.status = result.severity
        result.detail = f"Dilution level(s) not found: {missing}"
    else:
        result.detail = f"All required dilution levels found: {required}"
    return result


def check_standard_present(tabs, rule, result):
    da_df = tabs.get("data_analysis_parsed")
    expected = [float(c) for c in rule.get("expected_std_concentrations", [])]
    if not isinstance(da_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Data Analysis not available"
        return result
    std_rows = da_df[da_df["row_type"] == "std"]
    if std_rows.empty:
        result.status = result.severity; result.detail = "No STD rows found"
        return result
    found = pd.to_numeric(std_rows["dilution_or_std"], errors="coerce").dropna().tolist()
    missing = [c for c in expected if not any(abs(c - f) < 0.01 for f in found)]
    if missing:
        result.status = result.severity
        result.detail = f"Expected STD concentrations {missing} not found. Found: {found}"
    else:
        result.detail = f"300 µM standard confirmed (found: {found})"
    return result


def check_is_peak_area_cv(tabs, rule, result):
    da_df = tabs.get("data_analysis_parsed")
    threshold = float(rule.get("cv_threshold", 25.0))
    if not isinstance(da_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Data Analysis not available"
        return result
    failures = []
    for compound, grp in da_df.groupby("compound_id"):
        rep_rows = _get_true_replicates(grp)
        vals = pd.to_numeric(rep_rows["is_peak_area"], errors="coerce").dropna()
        if len(vals) < 2: continue
        cv = vals.std(ddof=1) / vals.mean() * 100 if vals.mean() != 0 else 0
        if cv > threshold:
            failures.append(f"{compound}: IS PA %CV={cv:.1f}%")
    if failures:
        result.status = result.severity
        result.detail = "IS peak area CV exceeds threshold: " + "; ".join(failures)
        result.affected_compounds = [f.split(":")[0] for f in failures]
    else:
        result.detail = f"All IS peak area %CV ≤ {threshold}%"
    return result


def check_blank_peak_area(tabs, rule, result):
    da_df = tabs.get("data_analysis_parsed")
    max_frac = float(rule.get("max_blank_fraction", 0.01))
    if not isinstance(da_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Data Analysis not available"
        return result
    failures = []
    for compound, grp in da_df.groupby("compound_id"):
        blank_pa = pd.to_numeric(grp[grp["row_type"]=="blank"]["analyte_peak_area"], errors="coerce").mean()
        sample_pa = pd.to_numeric(_get_true_replicates(grp)["analyte_peak_area"], errors="coerce").mean()
        if pd.isna(blank_pa) or pd.isna(sample_pa) or sample_pa == 0: continue
        ratio = blank_pa / sample_pa
        if ratio > max_frac:
            failures.append(f"{compound}: blank/sample={ratio:.3f} > {max_frac}")
    if failures:
        result.status = result.severity
        result.detail = "High blank signal: " + "; ".join(failures)
        result.affected_compounds = [f.split(":")[0] for f in failures]
    else:
        result.detail = f"All blank signals ≤ {max_frac*100:.0f}% of sample"
    return result


def check_signal_to_noise(tabs, rule, result):
    da_df = tabs.get("data_analysis_parsed")
    min_snr = float(rule.get("min_snr", 3.0))
    if not isinstance(da_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Data Analysis not available"
        return result
    failures = []
    for compound, grp in da_df.groupby("compound_id"):
        blank_pa  = pd.to_numeric(grp[grp["row_type"]=="blank"]["analyte_peak_area"], errors="coerce").mean()
        sample_pa = pd.to_numeric(_get_true_replicates(grp)["analyte_peak_area"], errors="coerce").mean()
        if pd.isna(blank_pa) or pd.isna(sample_pa) or blank_pa == 0: continue
        snr = sample_pa / blank_pa
        if snr < min_snr:
            failures.append(f"{compound}: S/N={snr:.1f}")
    if failures:
        result.status = result.severity
        result.detail = "Low S/N: " + "; ".join(failures)
        result.affected_compounds = [f.split(":")[0] for f in failures]
    else:
        result.detail = f"All S/N ≥ {min_snr}"
    return result


def check_solubility_cv(tabs, rule, result):
    da_df = tabs.get("data_analysis_parsed")
    threshold = float(rule.get("cv_threshold", 25.0))
    if not isinstance(da_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Data Analysis not available"
        return result
    failures = []
    for compound, grp in da_df.groupby("compound_id"):
        sols = _compute_solubilities(grp)
        if len(sols) < 2: continue
        import numpy as np
        cv = (pd.Series(sols).std(ddof=1) / pd.Series(sols).mean() * 100
              if pd.Series(sols).mean() != 0 else 0)
        if cv > threshold:
            failures.append(f"{compound}: %CV={cv:.1f}%")
    if failures:
        result.status = result.severity
        result.detail = "Solubility %CV exceeds threshold: " + "; ".join(failures)
        result.affected_compounds = [f.split(":")[0] for f in failures]
    else:
        result.detail = f"All solubility %CV ≤ {threshold}%"
    return result


def check_solubility_calculation(tabs, rule, result):
    da_df = tabs.get("data_analysis_parsed")
    tol = float(rule.get("tolerance_pct", 1.0))
    if not isinstance(da_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Data Analysis not available"
        return result
    failures = []
    for compound, grp in da_df.groupby("compound_id"):
        std_rows = grp[grp["row_type"] == "std"]
        rep_rows = _get_true_replicates(grp)
        std_ratio = pd.to_numeric(std_rows["ratio"], errors="coerce").mean()
        std_conc  = pd.to_numeric(std_rows["dilution_or_std"], errors="coerce").mean()
        dil = _get_dilution_from_blank(grp)
        if pd.isna(std_ratio) or std_ratio == 0 or pd.isna(std_conc): continue
        for _, r in rep_rows.iterrows():
            ratio = pd.to_numeric(r["ratio"], errors="coerce")
            d = dil or _extract_dilution_from_sample_id(r.get("sample_id"))
            cro_val = pd.to_numeric(r.get("calculated_value"), errors="coerce")
            if pd.isna(ratio) or d is None or pd.isna(cro_val): continue
            expected = (ratio / std_ratio) * std_conc * d
            diff_pct = abs(expected - cro_val) / cro_val * 100 if cro_val != 0 else 0
            if diff_pct > tol:
                failures.append(f"{compound} ({r['sample_id']}): exp={expected:.3f} CRO={cro_val:.3f} diff={diff_pct:.1f}%")
    if failures:
        result.status = result.severity
        result.detail = "Calculation mismatch: " + "; ".join(failures)
    else:
        result.detail = f"All solubility calculations verified within ±{tol}%"
    return result


def check_average_calculation(tabs, rule, result):
    da_df = tabs.get("data_analysis_parsed")
    tol = float(rule.get("tolerance_pct", 1.0))
    if not isinstance(da_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Data Analysis not available"
        return result
    failures = []
    for compound, grp in da_df.groupby("compound_id"):
        sols = _compute_solubilities(grp)
        if len(sols) < 2: continue
        expected_avg = sum(sols) / len(sols)
        avg_rows = grp[grp["row_type"] == "average"]
        if avg_rows.empty: continue
        cro_avg = pd.to_numeric(avg_rows["calculated_value"].iloc[0], errors="coerce")
        if pd.isna(cro_avg): continue
        diff_pct = abs(expected_avg - cro_avg) / cro_avg * 100 if cro_avg != 0 else 0
        if diff_pct > tol:
            failures.append(f"{compound}: exp={expected_avg:.3f} CRO={cro_avg:.3f} diff={diff_pct:.1f}%")
    if failures:
        result.status = result.severity
        result.detail = "Average mismatch: " + "; ".join(failures)
    else:
        result.detail = f"All average calculations verified within ±{tol}%"
    return result


def check_cv_calculation(tabs, rule, result):
    da_df = tabs.get("data_analysis_parsed")
    tol = float(rule.get("tolerance_pct", 1.0))
    if not isinstance(da_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Data Analysis not available"
        return result
    failures = []
    for compound, grp in da_df.groupby("compound_id"):
        sols = _compute_solubilities(grp)
        if len(sols) < 2: continue
        import numpy as np
        s = pd.Series(sols)
        computed_cv = s.std(ddof=1) / s.mean() * 100 if s.mean() != 0 else 0
        std_rows = grp[grp["row_type"] == "std"]
        cro_cv = pd.to_numeric(
            std_rows["calculated_value"].iloc[0] if len(std_rows) > 0 else None,
            errors="coerce"
        )
        if pd.isna(cro_cv): continue
        diff = abs(computed_cv - cro_cv)
        if diff > tol:
            failures.append(f"{compound}: computed={computed_cv:.2f}% CRO={cro_cv:.2f}% diff={diff:.2f}%")
    if failures:
        result.status = result.severity
        result.detail = "%CV mismatch: " + "; ".join(failures)
    else:
        result.detail = f"All %CV calculations verified within ±{tol}%"
    return result


def check_summary_vs_raw_solubility(tabs, rule, result):
    summary_df = tabs.get("summary")
    da_df = tabs.get("data_analysis_parsed")
    tol = float(rule.get("tolerance_pct", 1.0))
    if not isinstance(summary_df, pd.DataFrame) or not isinstance(da_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Tabs not available"
        return result
    failures = []
    for compound, grp in da_df.groupby("compound_id"):
        sols = _compute_solubilities(grp)
        if len(sols) < 1: continue
        derived_avg = sum(sols) / len(sols)
        row = summary_df[summary_df["compound_id"].str.strip() == compound.strip()]
        if row.empty: continue
        summary_sol = pd.to_numeric(row["solubility_um"].iloc[0], errors="coerce")
        if pd.isna(summary_sol): continue
        diff_pct = abs(derived_avg - summary_sol) / summary_sol * 100 if summary_sol != 0 else 0
        if diff_pct > tol:
            failures.append(f"{compound}: derived={derived_avg:.3f} summary={summary_sol:.3f} diff={diff_pct:.1f}%")
    if failures:
        result.status = result.severity
        result.detail = "Summary vs raw mismatch: " + "; ".join(failures)
        result.affected_compounds = [f.split(":")[0] for f in failures]
    else:
        result.detail = f"Summary values match raw data within ±{tol}%"
    return result


def check_control_range(tabs, rule, result):
    summary_df = tabs.get("summary")
    compound_name = rule["compound_name"]
    field_name = rule["field"]
    lo, hi = float(rule["min"]), float(rule["max"])
    if not isinstance(summary_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Summary tab not available"
        return result
    row = summary_df[summary_df["compound_id"].str.strip().str.lower() == compound_name.lower()]
    if row.empty:
        result.status = "warn"
        result.detail = f"'{compound_name}' not found in Summary"
        return result
    val = pd.to_numeric(row[field_name].iloc[0], errors="coerce")
    if pd.isna(val):
        result.status = result.severity; result.detail = f"{compound_name} value missing"
        return result
    if lo <= val <= hi:
        result.detail = f"{compound_name} = {val:.3f} µM — within [{lo}, {hi}]"
    else:
        result.status = result.severity
        result.detail = f"{compound_name} = {val:.3f} µM — OUTSIDE [{lo}, {hi}]"
        result.affected_compounds = [compound_name]
    return result


def check_upload_columns_present(tabs, rule, result):
    upload_df = tabs.get("upload_df")
    required = rule.get("required_columns", [])
    if not isinstance(upload_df, pd.DataFrame):
        result.status = "warn"; result.detail = "Upload DataFrame not yet generated"
        return result
    missing = [c for c in required if c not in upload_df.columns]
    if missing:
        result.status = result.severity; result.detail = f"Missing: {missing}"
    else:
        result.detail = "All required upload columns present"
    return result


def check_upload_test_compounds_only(tabs, rule, result):
    upload_df = tabs.get("upload_df")
    if not isinstance(upload_df, pd.DataFrame) or "Compound_ID" not in upload_df.columns:
        result.status = "warn"; result.detail = "Upload DataFrame not available"
        return result
    controls = ["Progesterone", "Diclofenac"]
    found = [c for c in controls if any(upload_df["Compound_ID"].str.strip().str.lower() == c.lower())]
    if found:
        result.status = result.severity; result.detail = f"Controls in upload: {found}"
    else:
        result.detail = "Upload contains only test compounds"
    return result


# ── Registry ──────────────────────────────────────────────────────────────────

REGISTRY: dict[str, Any] = {
    "manual":                       check_manual,
    "compound_name_consistency":    check_compound_name_consistency,
    "text_contains":                check_text_contains,
    "numeric_threshold":            check_numeric_threshold,
    "q1_mz_match":                  check_q1_mz_match,
    "dilution_levels_present":      check_dilution_levels_present,
    "standard_present":             check_standard_present,
    "is_peak_area_cv":              check_is_peak_area_cv,
    "blank_peak_area":              check_blank_peak_area,
    "signal_to_noise":              check_signal_to_noise,
    "solubility_cv":                check_solubility_cv,
    "solubility_calculation":       check_solubility_calculation,
    "average_calculation":          check_average_calculation,
    "cv_calculation":               check_cv_calculation,
    "summary_vs_raw_solubility":    check_summary_vs_raw_solubility,
    "control_range":                check_control_range,
    "upload_columns_present":       check_upload_columns_present,
    "upload_test_compounds_only":   check_upload_test_compounds_only,
}
