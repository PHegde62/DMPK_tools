"""
tests/unit/test_frontend.py
================================================================================
Unit tests for app/frontend.py helper functions.

These tests verify the pure-Python logic of the frontend module —
data transformations, HTML generation, error message handling — without
actually launching Streamlit or making HTTP calls.

All HTTP calls are mocked so these tests run offline with no backend.
"""

from __future__ import annotations

import json
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers under test (import lazily after mocking streamlit)
# ---------------------------------------------------------------------------

# Streamlit calls st.set_page_config at module level; we must mock 'st'
# before importing frontend, otherwise the import itself would fail.
import sys
import types

def _make_st_stub():
    st = types.ModuleType("streamlit")
    # Stub out everything frontend.py calls at import/module scope
    for attr in (
        "set_page_config", "session_state", "markdown", "text_area",
        "button", "columns", "sidebar", "spinner", "info", "warning",
        "error", "dataframe", "caption", "json", "expander", "rerun",
        "slider", "checkbox", "write",
    ):
        setattr(st, attr, MagicMock())
    st.session_state = {}
    return st


# Patch streamlit before any frontend import
sys.modules.setdefault("streamlit", _make_st_stub())

from app.frontend import (
    _build_metabolite_df,
    _call_predict,
    _call_render,
    _metric_card,
    _prob_badge,
    _phase_pill,
    _render_property_grid,
    _render_soft_spot_list,
    _svg_to_data_uri,
    EXAMPLES,
)


# ===========================================================================
# 1. EXAMPLES registry
# ===========================================================================

class TestExamples:
    def test_five_examples_defined(self):
        assert len(EXAMPLES) >= 5

    def test_each_example_has_smiles_and_desc(self):
        for name, data in EXAMPLES.items():
            assert "smiles" in data, f"{name} missing 'smiles'"
            assert "desc"   in data, f"{name} missing 'desc'"
            assert data["smiles"].strip(), f"{name} has empty SMILES"

    def test_aspirin_smiles_correct(self):
        assert EXAMPLES["Aspirin"]["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"

    def test_ibuprofen_smiles_correct(self):
        assert "Cc1ccc" in EXAMPLES["Ibuprofen"]["smiles"] or \
               "CC(C)Cc1ccc" in EXAMPLES["Ibuprofen"]["smiles"]


# ===========================================================================
# 2. HTML helpers
# ===========================================================================

class TestProbBadge:
    def test_high_prob_gets_prob_high_class(self):
        badge = _prob_badge(0.35)
        assert "prob-high" in badge

    def test_medium_prob_gets_prob_medium_class(self):
        badge = _prob_badge(0.15)
        assert "prob-medium" in badge

    def test_low_prob_gets_prob_low_class(self):
        badge = _prob_badge(0.02)
        assert "prob-low" in badge

    def test_boundary_high(self):
        assert "prob-high" in _prob_badge(0.25)      # exact boundary

    def test_boundary_medium(self):
        assert "prob-medium" in _prob_badge(0.08)    # exact lower boundary

    def test_contains_formatted_value(self):
        badge = _prob_badge(0.3456)
        assert "0.346" in badge   # 3 decimal places

    def test_returns_html_string(self):
        badge = _prob_badge(0.5)
        assert "<span" in badge and "</span>" in badge


class TestPhasePill:
    def test_phase1_gets_phase_1_class(self):
        pill = _phase_pill(1)
        assert "phase-1" in pill

    def test_phase2_gets_phase_2_class(self):
        pill = _phase_pill(2)
        assert "phase-2" in pill

    def test_returns_html(self):
        assert "<span" in _phase_pill(1)


class TestMetricCard:
    def test_value_appears_in_output(self):
        card = _metric_card(42, "Total")
        assert "42" in card

    def test_label_appears_in_output(self):
        card = _metric_card(7, "Soft Spots")
        assert "Soft Spots" in card

    def test_returns_html_div(self):
        card = _metric_card(0, "x")
        assert "<div" in card


class TestSvgToDataUri:
    def test_produces_data_uri(self):
        uri = _svg_to_data_uri("<svg></svg>")
        assert uri.startswith("data:image/svg+xml;base64,")

    def test_roundtrip(self):
        import base64
        original = "<svg><circle/></svg>"
        uri = _svg_to_data_uri(original)
        b64_part = uri.split(",", 1)[1]
        decoded = base64.b64decode(b64_part).decode()
        assert decoded == original


# ===========================================================================
# 3. _render_property_grid
# ===========================================================================

class TestRenderPropertyGrid:
    PARENT = {
        "molecular_formula": "C9H8O4",
        "molecular_weight":  180.16,
        "exact_mass":        180.0423,
        "logp":              1.19,
        "tpsa":              63.6,
        "num_hbd":           1,
        "num_hba":           4,
        "num_heavy_atoms":   13,
        "num_rotatable_bonds": 3,
    }

    def test_returns_html_string(self):
        html = _render_property_grid(self.PARENT)
        assert isinstance(html, str)
        assert "<div" in html

    def test_formula_present(self):
        html = _render_property_grid(self.PARENT)
        assert "C9H8O4" in html

    def test_mw_present(self):
        html = _render_property_grid(self.PARENT)
        assert "180.16" in html

    def test_missing_keys_dont_crash(self):
        # Partial data should not raise
        html = _render_property_grid({"molecular_formula": "CH4"})
        assert isinstance(html, str)


# ===========================================================================
# 4. _render_soft_spot_list
# ===========================================================================

class TestRenderSoftSpotList:
    SPOTS = [
        {"atom_index": 7, "atom_symbol": "C", "rule_name": "aromatic_C_unhindered",
         "score": 0.85, "smarts_match": "[cH]"},
        {"atom_index": 1, "atom_symbol": "C", "rule_name": "alpha_carbonyl_C",
         "score": 0.75, "smarts_match": "[CH2]"},
        {"atom_index": 4, "atom_symbol": "O", "rule_name": "phenolic_OH_site",
         "score": 0.68, "smarts_match": "[c;$([c][OH1])]"},
    ]

    def test_returns_html(self):
        html = _render_soft_spot_list(self.SPOTS)
        assert "<div" in html

    def test_all_atom_indices_present(self):
        html = _render_soft_spot_list(self.SPOTS)
        for spot in self.SPOTS:
            assert str(spot["atom_index"]) in html

    def test_scores_present(self):
        html = _render_soft_spot_list(self.SPOTS)
        assert "0.85" in html
        assert "0.75" in html

    def test_rule_names_present(self):
        html = _render_soft_spot_list(self.SPOTS)
        assert "aromatic" in html
        assert "carbonyl" in html

    def test_empty_list_returns_fallback(self):
        html = _render_soft_spot_list([])
        assert "No soft spots" in html or "identified" in html

    def test_rank_numbers_present(self):
        html = _render_soft_spot_list(self.SPOTS)
        assert ">1<" in html
        assert ">2<" in html
        assert ">3<" in html


# ===========================================================================
# 5. _build_metabolite_df
# ===========================================================================

class TestBuildMetaboliteDF:
    METS = [
        {"smiles": "OC(=O)c1ccccc1O",     "probability": 0.32, "phase": 1,
         "reaction_name": "aromatic_OH",   "molecular_weight": 138.12, "molecular_formula": "C7H6O3"},
        {"smiles": "CC(=O)Nc1ccc(O)cc1",  "probability": 0.18, "phase": 1,
         "reaction_name": "deacetylation", "molecular_weight": 151.16, "molecular_formula": "C8H9NO2"},
        {"smiles": "OC(=O)c1ccccc1OC(=O)O", "probability": 0.09, "phase": 2,
         "reaction_name": "glucuronidation","molecular_weight": 182.13, "molecular_formula": "C8H6O5"},
    ]

    def test_returns_dataframe(self):
        df = _build_metabolite_df(self.METS)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = _build_metabolite_df(self.METS)
        assert len(df) == 3

    def test_sorted_by_probability_descending(self):
        df = _build_metabolite_df(self.METS)
        probs = df["Prob."].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_required_columns_present(self):
        df = _build_metabolite_df(self.METS)
        for col in ("SMILES", "Prob.", "Phase", "Reaction", "MW (Da)", "Formula"):
            assert col in df.columns, f"Missing column: {col}"

    def test_index_starts_at_1(self):
        df = _build_metabolite_df(self.METS)
        assert df.index[0] == 1

    def test_empty_input_returns_empty_df(self):
        df = _build_metabolite_df([])
        assert len(df) == 0

    def test_mw_formatted_as_string(self):
        df = _build_metabolite_df(self.METS)
        # MW column should be a string with 2 decimal places
        mw_val = df.iloc[0]["MW (Da)"]
        assert isinstance(mw_val, str)
        assert "." in mw_val

    def test_missing_mw_shows_dash(self):
        mets = [{"smiles": "C", "probability": 0.5, "phase": 1,
                  "reaction_name": "test", "molecular_weight": None,
                  "molecular_formula": None}]
        df = _build_metabolite_df(mets)
        assert df.iloc[0]["MW (Da)"] == "—"


# ===========================================================================
# 6. _call_predict — mocked HTTP
# ===========================================================================

MOCK_PREDICT_RESPONSE = {
    "engine_version": "test-1.0",
    "elapsed_s": 0.5,
    "warnings": [],
    "parent": {
        "input_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "inchi": "InChI=1S/C9H8O4/...",
        "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        "molecular_formula": "C9H8O4",
        "molecular_weight": 180.16,
        "exact_mass": 180.0423,
        "num_heavy_atoms": 13,
        "num_rotatable_bonds": 3,
        "num_hbd": 0,
        "num_hba": 4,
        "tpsa": 63.6,
        "logp": 1.19,
        "num_rings": 1,
        "num_aromatic_rings": 1,
    },
    "metabolites": [
        {"smiles": "OC(=O)c1ccccc1O", "probability": 0.32, "phase": 1,
         "reaction_name": "aromatic_OH", "molecular_weight": 138.12,
         "molecular_formula": "C7H6O3", "svg": None},
    ],
    "soft_spots": [
        {"atom_index": 7, "atom_symbol": "C", "rule_name": "aromatic_C_unhindered",
         "score": 0.85, "smarts_match": "[cH]"},
    ],
    "metabolites_total": 1,
    "phase1_count": 1,
    "phase2_count": 0,
    "soft_spots_total": 1,
}


class TestCallPredict:

    def _mock_response(self, status_code: int, json_body=None, text: str = ""):
        mock = MagicMock()
        mock.status_code = status_code
        mock.is_success = (200 <= status_code < 300)
        mock.json.return_value = json_body or {}
        mock.text = text
        return mock

    def test_success_returns_dict_and_no_error(self):
        with patch("httpx.post", return_value=self._mock_response(200, MOCK_PREDICT_RESPONSE)):
            result, err = _call_predict("CC(=O)Oc1ccccc1C(=O)O")
        assert result is not None
        assert err is None
        assert result["parent"]["molecular_formula"] == "C9H8O4"

    def test_422_returns_none_and_error_message(self):
        body = {"detail": [{"msg": "Invalid SMILES string", "loc": ["body", "smiles"]}]}
        with patch("httpx.post", return_value=self._mock_response(422, body)):
            result, err = _call_predict("BAD_SMILES")
        assert result is None
        assert err is not None
        assert "422" in err or "Invalid" in err or "invalid" in err.lower()

    def test_500_returns_none_and_error_message(self):
        body = {"detail": "Internal server error"}
        with patch("httpx.post", return_value=self._mock_response(500, body, "server error")):
            result, err = _call_predict("C")
        assert result is None
        assert err is not None
        assert "500" in err

    def test_connect_error_returns_message(self):
        import httpx as _httpx
        with patch("httpx.post", side_effect=_httpx.ConnectError("refused")):
            result, err = _call_predict("C")
        assert result is None
        assert err is not None
        assert "refused" in err.lower() or "connection" in err.lower()

    def test_timeout_returns_message(self):
        import httpx as _httpx
        with patch("httpx.post", side_effect=_httpx.TimeoutException("timeout")):
            result, err = _call_predict("C")
        assert result is None
        assert err is not None
        assert "timed out" in err.lower() or "timeout" in err.lower()

    def test_422_with_string_detail(self):
        body = {"detail": "SMILES parse error"}
        with patch("httpx.post", return_value=self._mock_response(422, body)):
            result, err = _call_predict("BAD")
        assert result is None
        assert "SMILES parse error" in err or "422" in err


# ===========================================================================
# 7. _call_render — mocked HTTP
# ===========================================================================

SAMPLE_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="560" height="380"><rect/></svg>'


class TestCallRender:

    def _mock_response(self, status_code: int, text: str = ""):
        mock = MagicMock()
        mock.status_code = status_code
        mock.is_success = (200 <= status_code < 300)
        mock.text = text
        return mock

    def test_success_returns_svg_string(self):
        with patch("httpx.post", return_value=self._mock_response(200, SAMPLE_SVG)):
            svg, err = _call_render("CC(=O)Oc1ccccc1C(=O)O", [7, 1])
        assert svg == SAMPLE_SVG
        assert err is None

    def test_svg_contains_svg_tag(self):
        with patch("httpx.post", return_value=self._mock_response(200, SAMPLE_SVG)):
            svg, err = _call_render("C", [])
        assert "<svg" in svg

    def test_error_returns_none_svg(self):
        with patch("httpx.post", return_value=self._mock_response(500, "server error")):
            svg, err = _call_render("C", [])
        assert svg is None
        assert err is not None

    def test_connect_error_returns_message(self):
        import httpx as _httpx
        with patch("httpx.post", side_effect=_httpx.ConnectError("refused")):
            svg, err = _call_render("C", [])
        assert svg is None
        assert err is not None
        assert "refused" in err.lower() or "unreachable" in err.lower()

    def test_timeout_returns_message(self):
        import httpx as _httpx
        with patch("httpx.post", side_effect=_httpx.TimeoutException("t/o")):
            svg, err = _call_render("C", [])
        assert svg is None
        assert "timed out" in err.lower() or "timeout" in err.lower()

    def test_default_highlight_color_is_coral(self):
        """When no color is passed the default coral is used (checked via call args)."""
        captured = {}
        def mock_post(url, json=None, timeout=None):
            captured["json"] = json
            return self._mock_response(200, SAMPLE_SVG)

        with patch("httpx.post", side_effect=mock_post):
            _call_render("C", [0])

        color = captured["json"]["highlight_color"]
        assert color[0] == 1.0   # R = 1.0 (red component of coral)
        assert len(color) == 4   # RGBA
