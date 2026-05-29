"""
test_qc_engine.py
Unit tests for rule_evaluator and qc_engine.
Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.rule_evaluator import (
    evaluate_rule, check_control_range, check_numeric_threshold,
    check_blank_peak_area, check_signal_to_noise, check_is_peak_area_cv,
    QCResult,
)
from modules.qc_engine import run_qc, qc_summary


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_summary_df(progesterone_val=12.0, diclofenac_val=277.0):
    return pd.DataFrame({
        "compound_id": ["Progesterone", "Diclofenac", "GEN-0072267"],
        "batch_id":    ["-", "-", "CF-001"],
        "solubility_um": [progesterone_val, diclofenac_val, 273.08],
    })


def make_bioanalytical_df():
    return pd.DataFrame({
        "compound_id": ["Progesterone", "Diclofenac", "GEN-0072267"],
        "q1_mz":       [315.3, 296.0, 868.634],
        "dp_v":        [106.0, 80.0, 85.0],
        "ce_v":        [35.0, 30.0, 80.0],
    })


def make_materials_df():
    return pd.DataFrame({
        "compound_id": ["GEN-0072267"],
        "mw_gmol":     [867.44],
    })


def make_da_df(prog_is_pa=None, gen_is_pa=None):
    """Minimal Data Analysis DataFrame for IS peak area tests."""
    rows = []
    # Progesterone block
    for pa in (prog_is_pa or [642000, 677000, 696000]):
        rows.append({
            "compound_id": "Progesterone",
            "row_type": "replicate",
            "is_peak_area": pa,
            "analyte_peak_area": 185000,
            "ratio": 0.273,
            "sample_id": "Buffer-Sample-1000",
        })
    # GEN block
    for pa in (gen_is_pa or [603000, 574000, 597000]):
        rows.append({
            "compound_id": "GEN-0072267",
            "row_type": "replicate",
            "is_peak_area": pa,
            "analyte_peak_area": 3380000,
            "ratio": 5.6,
            "sample_id": "Buffer-Sample-100",
        })
    # Blank rows
    rows.append({
        "compound_id": "Progesterone", "row_type": "blank",
        "is_peak_area": 650000, "analyte_peak_area": 5960,
        "ratio": 0.009, "sample_id": "Buffer-Single-Blank",
    })
    rows.append({
        "compound_id": "GEN-0072267", "row_type": "blank",
        "is_peak_area": 671000, "analyte_peak_area": 0,
        "ratio": 0.0, "sample_id": "Buffer-Single-Blank",
    })
    return pd.DataFrame(rows)


# ── Tests: control range ──────────────────────────────────────────────────────

class TestControlRange:
    def _rule(self, compound, lo, hi):
        return {
            "id": "SUM_003", "section": "Summary", "description": "test",
            "check": "control_range",
            "compound_name": compound,
            "field": "solubility_um",
            "min": lo, "max": hi,
            "severity": "fail",
        }

    def test_progesterone_pass(self):
        tabs = {"summary": make_summary_df()}
        rule = self._rule("Progesterone", 9.8, 16.3)
        result = evaluate_rule(rule, tabs)
        assert result.status == "pass"

    def test_progesterone_fail_low(self):
        tabs = {"summary": make_summary_df(progesterone_val=5.0)}
        rule = self._rule("Progesterone", 9.8, 16.3)
        result = evaluate_rule(rule, tabs)
        assert result.status == "fail"

    def test_diclofenac_pass(self):
        tabs = {"summary": make_summary_df()}
        rule = self._rule("Diclofenac", 225.0, 375.0)
        result = evaluate_rule(rule, tabs)
        assert result.status == "pass"

    def test_diclofenac_fail_high(self):
        tabs = {"summary": make_summary_df(diclofenac_val=400.0)}
        rule = self._rule("Diclofenac", 225.0, 375.0)
        result = evaluate_rule(rule, tabs)
        assert result.status == "fail"
        assert "Diclofenac" in result.affected_compounds

    def test_missing_control(self):
        tabs = {"summary": pd.DataFrame({"compound_id": ["GEN-0072267"], "solubility_um": [273.0]})}
        rule = self._rule("Progesterone", 9.8, 16.3)
        result = evaluate_rule(rule, tabs)
        assert result.status == "warn"  # can't fail what isn't there


# ── Tests: numeric threshold ──────────────────────────────────────────────────

class TestNumericThreshold:
    def _rule(self, field, op, threshold):
        return {
            "id": "BIO_004", "section": "Bioanalytical", "description": "test",
            "check": "numeric_threshold",
            "tab": "bioanalytical", "field": field,
            "operator": op, "threshold": threshold,
            "severity": "fail",
        }

    def test_dp_all_pass(self):
        tabs = {"bioanalytical": make_bioanalytical_df()}
        result = evaluate_rule(self._rule("dp_v", ">=", 30), tabs)
        assert result.status == "pass"

    def test_dp_one_fail(self):
        df = make_bioanalytical_df()
        df.loc[0, "dp_v"] = 20.0   # Progesterone DP below 30
        tabs = {"bioanalytical": df}
        result = evaluate_rule(self._rule("dp_v", ">=", 30), tabs)
        assert result.status == "fail"

    def test_ce_all_pass(self):
        tabs = {"bioanalytical": make_bioanalytical_df()}
        result = evaluate_rule(self._rule("ce_v", "<=", 100), tabs)
        assert result.status == "pass"


# ── Tests: IS peak area CV ────────────────────────────────────────────────────

class TestISPeakAreaCV:
    def _rule(self):
        return {
            "id": "RAW_002", "section": "Raw Data", "description": "test",
            "check": "is_peak_area_cv", "tab": "data_analysis",
            "cv_threshold": 25.0, "severity": "fail",
        }

    def test_low_cv_passes(self):
        tabs = {"data_analysis_parsed": make_da_df()}
        result = evaluate_rule(self._rule(), tabs)
        assert result.status == "pass"

    def test_high_cv_fails(self):
        # IS PA values with very high variance
        tabs = {"data_analysis_parsed": make_da_df(prog_is_pa=[100000, 900000, 500000])}
        result = evaluate_rule(self._rule(), tabs)
        assert result.status == "fail"
        assert "Progesterone" in result.affected_compounds


# ── Tests: blank peak area ────────────────────────────────────────────────────

class TestBlankPeakArea:
    def _rule(self):
        return {
            "id": "RAW_004", "section": "Raw Data", "description": "test",
            "check": "blank_peak_area", "tab": "data_analysis",
            "blank_row_label": "Blank",
            "analyte_field": "analyte_peak_area",
            "max_blank_fraction": 0.01, "severity": "warn",
        }

    def test_gen_blank_zero_passes(self):
        """GEN-0072267 blank analyte PA = 0 — should pass (zero noise)."""
        # GEN has blank PA = 0, so ratio = 0 — skip (can't divide by zero sample)
        # Progesterone blank (5960) / sample (185000) = 3.2% > 1% → warns
        # This is correct scientific behaviour — test the warn is on Progesterone, not GEN
        tabs = {"data_analysis_parsed": make_da_df()}
        result = evaluate_rule(self._rule(), tabs)
        assert result.status == "warn"
        assert "Progesterone" in result.detail
        assert "GEN-0072267" not in result.detail

    def test_clean_blank_passes(self):
        """If blank is well below threshold, no warning."""
        da = make_da_df()
        # Set Progesterone blank PA very low
        da.loc[da["compound_id"] == "Progesterone", "analyte_peak_area"] = (
            da.loc[da["compound_id"] == "Progesterone", "analyte_peak_area"]
            .where(da.loc[da["compound_id"] == "Progesterone", "row_type"] != "blank", 100)
        )
        tabs = {"data_analysis_parsed": da}
        result = evaluate_rule(self._rule(), tabs)
        assert result.status == "pass"


# ── Tests: qc_summary ────────────────────────────────────────────────────────

class TestQCSummary:
    def _make_result(self, status, severity="fail"):
        return QCResult(
            rule_id="X001", section="Test", description="test",
            status=status, severity=severity,
        )

    def test_upload_blocked_on_fail(self):
        results = [
            self._make_result("pass"),
            self._make_result("fail"),
            self._make_result("warn"),
        ]
        s = qc_summary(results)
        assert s["upload_blocked"] is True
        assert s["failed"] == 1

    def test_upload_not_blocked_all_pass(self):
        results = [self._make_result("pass"), self._make_result("warn")]
        s = qc_summary(results)
        assert s["upload_blocked"] is False

    def test_counts(self):
        results = [
            self._make_result("pass"),
            self._make_result("pass"),
            self._make_result("fail"),
            self._make_result("warn"),
            self._make_result("manual"),
        ]
        s = qc_summary(results)
        assert s == {"total": 5, "passed": 2, "failed": 1, "warned": 1,
                     "manual": 1, "upload_blocked": True}
