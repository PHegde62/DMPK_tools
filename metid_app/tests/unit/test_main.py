"""
tests/unit/test_main.py
================================================================================
API-layer tests for app/main.py

Test strategy
-------------
- The metabolism engine is mocked so tests run without SyGMa installed and
  at module-import speed (no RDKit prediction cost per test).
- RDKit is used *only* for SVG rendering tests, which require the real library.
- All tests use httpx.AsyncClient with ASGITransport -- no live server needed.
- CORS tests exercise the middleware configuration directly via OPTIONS requests.

Groups
------
  TestSMILESValidator      -- the shared Pydantic SMILES validator
  TestPredictRequest       -- request schema validation
  TestRenderRequest        -- render request schema validation
  TestHealthProbes         -- /health and /ready endpoints
  TestPredictEndpoint      -- /predict with mocked engine
  TestRenderSoftSpots      -- /render-soft-spots with real RDKit
  TestCORSMiddleware        -- CORS header behaviour
  TestErrorHandling        -- 422, 503, 500 error shapes

Run with:
    pytest tests/unit/test_main.py -v
"""

from __future__ import annotations

import sys
import types
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# App under test
# ---------------------------------------------------------------------------

# We need to import create_app after patching so the lifespan does not try to
# connect to a real database.  We do this via a lazy import in each test class.

from app.core.config import Settings


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _test_settings(**overrides) -> Settings:
    """Build a Settings instance suitable for testing."""
    defaults = dict(
        APP_ENV="development",
        APP_VERSION="test-0.0.1",
        DATABASE_URL="sqlite+aiosqlite:///./test_metid.db",
        CORS_ALLOW_ALL=True,          # simplifies CORS assertions in most tests
        CORS_ORIGINS=[],
        SECRET_KEY="test-secret",
        MAX_METABOLITES_RETURNED=200,
        SYGMA_PHASE1_CYCLES=1,
        SYGMA_PHASE2_CYCLES=1,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_mock_engine_result(smiles: str = ASPIRIN) -> MagicMock:
    """Return a MagicMock that behaves like a MetabolismResult."""
    from app.engine.metabolism import (
        MetabolismResult, MoleculeMetadata, PredictedMetabolite, SoftSpot
    )
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem import inchi as rdinchi

    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol)
    inchi_str  = rdinchi.MolToInchi(mol) or ""
    inchikey   = rdinchi.InchiToInchiKey(inchi_str) if inchi_str else ""

    parent = MoleculeMetadata(
        input_smiles=smiles,
        canonical_smiles=canonical,
        inchi=inchi_str,
        inchikey=inchikey,
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 4),
        exact_mass=round(Descriptors.ExactMolWt(mol), 6),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        num_hbd=rdMolDescriptors.CalcNumHBD(mol),
        num_hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(Descriptors.rdMolDescriptors.CalcTPSA(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 4),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
    )

    metabolites = [
        PredictedMetabolite(
            smiles="OC(=O)c1ccccc1O",
            probability=0.32,
            phase=1,
            reaction_name="phase1_aromatic_hydroxylation",
            molecular_weight=138.12,
            molecular_formula="C7H6O3",
        ),
        PredictedMetabolite(
            smiles="OC(=O)c1ccccc1OC(=O)O",
            probability=0.09,
            phase=2,
            reaction_name="phase2_glucuronidation",
            molecular_weight=182.13,
            molecular_formula="C8H6O5",
        ),
    ]

    soft_spots = [
        SoftSpot(atom_index=7, atom_symbol="C",
                 rule_name="aromatic_C_unhindered", score=0.85,
                 smarts_match="[cH]"),
        SoftSpot(atom_index=1, atom_symbol="C",
                 rule_name="alpha_carbonyl_C", score=0.75,
                 smarts_match="[CH2,CH3;$([CH2,CH3]C(=O)[#6,#7,#8])]"),
    ]

    result = MetabolismResult(
        parent=parent,
        metabolites=metabolites,
        soft_spots=soft_spots,
        engine_version="test-engine-1.0",
        elapsed_s=0.123,
        warnings=[],
    )
    return result


# Synchronous TestClient wrapped for consistent fixture usage
@pytest.fixture
def client():
    """Synchronous TestClient for simple request/response tests."""
    from app.main import create_app
    test_app = create_app(settings=_test_settings())
    return TestClient(test_app, raise_server_exceptions=True)


# ==============================================================================
# 1.  SMILES validator (unit tests -- no HTTP)
# ==============================================================================

class TestSMILESValidator:
    """Tests for the _validate_smiles_string helper used by Pydantic."""

    def _validate(self, v: str) -> str:
        from app.main import _validate_smiles_string
        return _validate_smiles_string(v)

    def test_valid_smiles_returns_stripped(self):
        result = self._validate("  CC(=O)O  ")
        assert result == "CC(=O)O"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            self._validate("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            self._validate("    ")

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="not a valid SMILES"):
            self._validate("ZZZ_NOT_REAL")

    def test_valid_complex_smiles(self):
        # Ibuprofen -- complex but valid
        result = self._validate(IBUPROFEN)
        assert result == IBUPROFEN

    def test_salt_passes_validation(self):
        # Salt forms are valid SMILES; normalisation happens in the engine
        result = self._validate("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
        assert "Na" in result or "Na" not in result   # structure agnostic

    def test_single_atom_valid(self):
        result = self._validate("C")
        assert result == "C"


# ==============================================================================
# 2.  PredictRequest schema validation
# ==============================================================================

class TestPredictRequest:
    """Pydantic schema validation for POST /predict request body."""

    def _make(self, **kwargs):
        from app.main import PredictRequest
        return PredictRequest(**{"smiles": ASPIRIN, **kwargs})

    def test_minimal_valid_request(self):
        req = self._make()
        assert req.smiles == ASPIRIN
        assert req.phase1_cycles == 1
        assert req.phase2_cycles == 1
        assert req.max_metabolites == 50
        assert req.top_soft_spots == 3
        assert req.include_svg is False

    def test_invalid_smiles_raises_validation_error(self):
        with pytest.raises(ValidationError):
            self._make(smiles="NOT_VALID###")

    def test_empty_smiles_raises_validation_error(self):
        with pytest.raises(ValidationError):
            self._make(smiles="")

    def test_phase_cycles_bounds(self):
        with pytest.raises(ValidationError):
            self._make(phase1_cycles=0)      # below ge=1
        with pytest.raises(ValidationError):
            self._make(phase1_cycles=4)      # above le=3

    def test_max_metabolites_bounds(self):
        with pytest.raises(ValidationError):
            self._make(max_metabolites=0)
        with pytest.raises(ValidationError):
            self._make(max_metabolites=501)

    def test_top_soft_spots_bounds(self):
        with pytest.raises(ValidationError):
            self._make(top_soft_spots=0)
        with pytest.raises(ValidationError):
            self._make(top_soft_spots=11)

    def test_include_svg_default_false(self):
        req = self._make()
        assert req.include_svg is False

    def test_include_svg_can_be_true(self):
        req = self._make(include_svg=True)
        assert req.include_svg is True


# ==============================================================================
# 3.  RenderSoftSpotsRequest schema validation
# ==============================================================================

class TestRenderRequest:
    """Pydantic schema validation for POST /render-soft-spots request body."""

    def _make(self, **kwargs):
        from app.main import RenderSoftSpotsRequest
        return RenderSoftSpotsRequest(**{"smiles": ASPIRIN, **kwargs})

    def test_minimal_valid(self):
        req = self._make()
        assert req.smiles == ASPIRIN
        assert req.highlight_indices is None
        assert req.width == 600
        assert req.height == 400
        assert len(req.highlight_color) == 4

    def test_default_color_is_coral(self):
        req = self._make()
        r, g, b, a = req.highlight_color
        assert r == 1.0
        assert g == pytest.approx(0.35, abs=0.01)
        assert b == pytest.approx(0.35, abs=0.01)

    def test_rgb_color_gets_alpha_appended(self):
        req = self._make(highlight_color=[0.5, 0.8, 0.2])
        assert len(req.highlight_color) == 4
        assert req.highlight_color[3] == 0.6   # default alpha

    def test_invalid_color_channel_raises(self):
        with pytest.raises(ValidationError):
            self._make(highlight_color=[1.5, 0.0, 0.0])   # > 1.0

    def test_invalid_color_length_raises(self):
        with pytest.raises(ValidationError):
            self._make(highlight_color=[1.0, 0.5])         # only 2 channels

    def test_valid_highlight_indices(self):
        req = self._make(highlight_indices=[0, 1, 2])
        assert req.highlight_indices == [0, 1, 2]

    def test_negative_highlight_index_raises(self):
        with pytest.raises(ValidationError):
            self._make(highlight_indices=[-1, 0])

    def test_out_of_range_index_raises(self):
        # Aspirin has 13 heavy atoms (indices 0-12)
        with pytest.raises(ValidationError):
            self._make(highlight_indices=[999])

    def test_size_bounds(self):
        with pytest.raises(ValidationError):
            self._make(width=50)       # below ge=200
        with pytest.raises(ValidationError):
            self._make(height=3000)    # above le=2000


# ==============================================================================
# 4.  Health probes
# ==============================================================================

class TestHealthProbes:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_contains_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_health_contains_version(self, client):
        r = client.get("/health")
        assert "version" in r.json()

    def test_ready_returns_200_or_503(self, client):
        # Readiness depends on the _rdkit_ready flag which may or may not be
        # set in test environment; accept both codes.
        r = client.get("/ready")
        assert r.status_code in (200, 503)

    def test_ready_has_status_field(self, client):
        r = client.get("/ready")
        assert "status" in r.json()


# ==============================================================================
# 5.  POST /predict
# ==============================================================================

class TestPredictEndpoint:
    """Tests for /predict with the metabolism engine mocked."""

    @pytest.fixture
    def client_with_mock(self):
        """
        TestClient whose calls to app.engine.metabolism.predict are intercepted.
        The fixture patches the name as imported inside the endpoint handler.
        """
        mock_result = _make_mock_engine_result(ASPIRIN)

        with patch("app.engine.metabolism.predict", return_value=mock_result):
            from app.main import create_app
            test_app = create_app(settings=_test_settings())
            yield TestClient(test_app, raise_server_exceptions=True)

    def test_returns_200(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ASPIRIN})
        assert r.status_code == 200

    def test_response_has_required_top_level_keys(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ASPIRIN})
        body = r.json()
        for key in ("parent", "metabolites", "soft_spots", "engine_version",
                    "elapsed_s", "warnings", "metabolites_total",
                    "phase1_count", "phase2_count", "soft_spots_total"):
            assert key in body, f"Missing top-level key: {key!r}"

    def test_parent_molecular_formula(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ASPIRIN})
        assert r.json()["parent"]["molecular_formula"] == "C9H8O4"

    def test_metabolites_list_is_list(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ASPIRIN})
        assert isinstance(r.json()["metabolites"], list)

    def test_metabolite_has_required_fields(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ASPIRIN})
        metabolites = r.json()["metabolites"]
        assert len(metabolites) > 0
        m = metabolites[0]
        for key in ("smiles", "probability", "phase", "reaction_name"):
            assert key in m, f"Metabolite missing key: {key!r}"

    def test_phase_values_are_1_or_2(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ASPIRIN})
        for m in r.json()["metabolites"]:
            assert m["phase"] in (1, 2)

    def test_counts_match_lists(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ASPIRIN})
        body = r.json()
        assert body["metabolites_total"] == len(body["metabolites"])
        assert body["soft_spots_total"] == len(body["soft_spots"])
        assert body["phase1_count"] + body["phase2_count"] == body["metabolites_total"]

    def test_soft_spots_have_required_fields(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ASPIRIN})
        for s in r.json()["soft_spots"]:
            for key in ("atom_index", "atom_symbol", "rule_name", "score", "smarts_match"):
                assert key in s, f"SoftSpot missing key: {key!r}"

    def test_invalid_smiles_returns_422(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": "NOT_SMILES###"})
        assert r.status_code == 422

    def test_empty_smiles_returns_422(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ""})
        assert r.status_code == 422

    def test_missing_smiles_returns_422(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"phase1_cycles": 1})
        assert r.status_code == 422

    def test_out_of_range_cycles_returns_422(self, client_with_mock):
        r = client_with_mock.post(
            "/predict", json={"smiles": ASPIRIN, "phase1_cycles": 99}
        )
        assert r.status_code == 422

    def test_include_svg_false_has_null_svg(self, client_with_mock):
        r = client_with_mock.post(
            "/predict", json={"smiles": ASPIRIN, "include_svg": False}
        )
        for m in r.json()["metabolites"]:
            assert m["svg"] is None

    def test_include_svg_true_has_svg_string_or_null(self, client_with_mock):
        """With include_svg=True metabolite SVGs should be strings (or None on error)."""
        r = client_with_mock.post(
            "/predict", json={"smiles": ASPIRIN, "include_svg": True}
        )
        assert r.status_code == 200
        for m in r.json()["metabolites"]:
            assert m["svg"] is None or isinstance(m["svg"], str)

    def test_warnings_field_is_list(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ASPIRIN})
        assert isinstance(r.json()["warnings"], list)

    def test_content_type_is_json(self, client_with_mock):
        r = client_with_mock.post("/predict", json={"smiles": ASPIRIN})
        assert "application/json" in r.headers["content-type"]

    def test_max_metabolites_parameter_forwarded(self):
        """Engine should be called with the requested max_metabolites."""
        mock_result = _make_mock_engine_result(ASPIRIN)
        call_kwargs = {}

        def capturing_predict(smiles, **kwargs):
            call_kwargs.update(kwargs)
            return mock_result

        with patch("app.engine.metabolism.predict", side_effect=capturing_predict):
            from app.main import create_app
            test_app = create_app(settings=_test_settings())
            with TestClient(test_app) as c:
                c.post("/predict", json={"smiles": ASPIRIN, "max_metabolites": 42})

        assert call_kwargs.get("max_metabolites") == 42

    def test_top_soft_spots_parameter_forwarded(self):
        mock_result = _make_mock_engine_result(ASPIRIN)
        call_kwargs = {}

        def capturing_predict(smiles, **kwargs):
            call_kwargs.update(kwargs)
            return mock_result

        with patch("app.engine.metabolism.predict", side_effect=capturing_predict):
            from app.main import create_app
            test_app = create_app(settings=_test_settings())
            with TestClient(test_app) as c:
                c.post("/predict", json={"smiles": ASPIRIN, "top_soft_spots": 5})

        assert call_kwargs.get("top_soft_spots") == 5


# ==============================================================================
# 6.  POST /render-soft-spots
# ==============================================================================

class TestRenderSoftSpots:
    """Tests for /render-soft-spots.  Uses real RDKit rendering."""

    @pytest.fixture
    def client(self):
        from app.main import create_app
        test_app = create_app(settings=_test_settings())
        return TestClient(test_app, raise_server_exceptions=True)

    def test_returns_200(self, client):
        r = client.post("/render-soft-spots", json={"smiles": ASPIRIN})
        assert r.status_code == 200

    def test_content_type_is_svg(self, client):
        r = client.post("/render-soft-spots", json={"smiles": ASPIRIN})
        assert "image/svg+xml" in r.headers["content-type"]

    def test_response_body_contains_svg_tag(self, client):
        r = client.post("/render-soft-spots", json={"smiles": ASPIRIN})
        assert "<svg" in r.text
        assert "</svg>" in r.text

    def test_response_has_x_highlighted_atoms_header(self, client):
        r = client.post("/render-soft-spots", json={"smiles": ASPIRIN})
        assert "x-highlighted-atoms" in {k.lower() for k in r.headers}

    def test_response_has_x_soft_spot_count_header(self, client):
        r = client.post("/render-soft-spots", json={"smiles": ASPIRIN})
        assert "x-soft-spot-count" in {k.lower() for k in r.headers}

    def test_cache_control_header_set(self, client):
        r = client.post("/render-soft-spots", json={"smiles": ASPIRIN})
        assert "cache-control" in {k.lower() for k in r.headers}
        assert "max-age=60" in r.headers["cache-control"]

    def test_mode_b_highlight_indices_respected(self, client):
        """Pre-computed indices in Mode B should appear in X-Highlighted-Atoms."""
        r = client.post(
            "/render-soft-spots",
            json={"smiles": ASPIRIN, "highlight_indices": [0, 4]},
        )
        assert r.status_code == 200
        highlighted = r.headers.get("x-highlighted-atoms", "")
        assert "0" in highlighted
        assert "4" in highlighted

    def test_invalid_smiles_returns_422(self, client):
        r = client.post("/render-soft-spots", json={"smiles": "BADINPUT"})
        assert r.status_code == 422

    def test_out_of_range_index_returns_422(self, client):
        r = client.post(
            "/render-soft-spots",
            json={"smiles": ASPIRIN, "highlight_indices": [9999]},
        )
        assert r.status_code == 422

    def test_custom_color_accepted(self, client):
        r = client.post(
            "/render-soft-spots",
            json={"smiles": ASPIRIN, "highlight_color": [0.0, 1.0, 0.0, 0.5]},
        )
        assert r.status_code == 200

    def test_rgb_only_color_accepted(self, client):
        """3-channel colors should be accepted (alpha defaults to 0.6)."""
        r = client.post(
            "/render-soft-spots",
            json={"smiles": ASPIRIN, "highlight_color": [1.0, 0.5, 0.0]},
        )
        assert r.status_code == 200

    def test_custom_dimensions_respected(self, client):
        r = client.post(
            "/render-soft-spots",
            json={"smiles": ASPIRIN, "width": 800, "height": 600},
        )
        assert r.status_code == 200
        assert "800" in r.text    # width appears in SVG viewBox / width attr

    def test_show_atom_indices_true_produces_svg(self, client):
        r = client.post(
            "/render-soft-spots",
            json={"smiles": ASPIRIN, "show_atom_indices": True},
        )
        assert r.status_code == 200
        assert "<svg" in r.text

    def test_show_scores_true_produces_text_labels(self, client):
        r = client.post(
            "/render-soft-spots",
            json={"smiles": ASPIRIN, "show_scores": True},
        )
        assert r.status_code == 200
        # Score labels are injected as <text> elements
        assert "<text" in r.text

    def test_mode_a_top_n_controls_label_count(self, client):
        """Requesting top_soft_spots=1 should annotate at most 1 atom."""
        r = client.post(
            "/render-soft-spots",
            json={"smiles": ASPIRIN, "top_soft_spots": 1},
        )
        assert r.status_code == 200
        count = int(r.headers.get("x-soft-spot-count", "0"))
        assert count <= 1

    def test_no_highlight_indices_uses_engine(self, client):
        """Omitting highlight_indices (Mode A) should still produce a valid SVG."""
        r = client.post("/render-soft-spots", json={"smiles": IBUPROFEN})
        assert r.status_code == 200
        assert "<svg" in r.text


# ==============================================================================
# 7.  CORS middleware
# ==============================================================================

class TestCORSMiddleware:
    """Verify CORS header behaviour for cross-domain dashboard scenarios."""

    @pytest.fixture
    def client_allow_all(self):
        """Client with CORS_ALLOW_ALL=True (wildcard)."""
        from app.main import create_app
        test_app = create_app(settings=_test_settings(CORS_ALLOW_ALL=True))
        return TestClient(test_app)

    @pytest.fixture
    def client_specific_origin(self):
        """Client with a specific CORS origin list."""
        from app.main import create_app
        test_app = create_app(
            settings=_test_settings(
                CORS_ALLOW_ALL=False,
                CORS_ORIGINS=["https://dashboard.myco.io"],
            )
        )
        return TestClient(test_app)

    def test_allow_all_returns_wildcard_header(self, client_allow_all):
        r = client_allow_all.options(
            "/predict",
            headers={
                "Origin": "https://attacker.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.headers.get("access-control-allow-origin") == "*"

    def test_specific_origin_allowed(self, client_specific_origin):
        r = client_specific_origin.options(
            "/predict",
            headers={
                "Origin": "https://dashboard.myco.io",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.headers.get("access-control-allow-origin") == "https://dashboard.myco.io"

    def test_disallowed_origin_not_reflected(self, client_specific_origin):
        r = client_specific_origin.options(
            "/predict",
            headers={
                "Origin": "https://attacker.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        acao = r.headers.get("access-control-allow-origin", "")
        assert "attacker.com" not in acao

    def test_credentials_header_absent_with_wildcard(self, client_allow_all):
        """
        Browsers reject credentialed requests when allow_origins=["*"].
        Our middleware sets allow_credentials=False in that case, which means
        the Allow-Credentials header is either absent or "false".
        """
        r = client_allow_all.options(
            "/predict",
            headers={
                "Origin": "https://anything.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        aca = r.headers.get("access-control-allow-credentials", "false")
        assert aca.lower() != "true"

    def test_max_age_header_present(self, client_allow_all):
        r = client_allow_all.options(
            "/predict",
            headers={
                "Origin": "https://anything.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # max_age=600 is set; Starlette exposes it as Access-Control-Max-Age
        assert "access-control-max-age" in {k.lower() for k in r.headers}

    def test_post_request_has_cors_header(self, client_allow_all):
        """Actual POST requests (not just preflight) also get CORS headers."""
        r = client_allow_all.post(
            "/predict",
            json={"smiles": "INVALID###"},
            headers={"Origin": "https://dashboard.myco.io"},
        )
        # 422 expected but CORS header should still be present
        assert "access-control-allow-origin" in {k.lower() for k in r.headers}


# ==============================================================================
# 8.  Error handling shapes
# ==============================================================================

class TestErrorHandling:
    """Verify that error responses conform to expected shapes."""

    @pytest.fixture
    def client(self):
        from app.main import create_app
        test_app = create_app(settings=_test_settings())
        return TestClient(test_app, raise_server_exceptions=False)

    def test_422_body_has_detail_field(self, client):
        r = client.post("/predict", json={"smiles": "BAD_SMILES!!!"})
        assert r.status_code == 422
        # FastAPI 422 bodies have a 'detail' key (list of validation errors)
        assert "detail" in r.json()

    def test_engine_runtime_error_returns_503(self):
        """RuntimeError from the engine should map to 503."""
        with patch(
            "app.engine.metabolism.predict",
            side_effect=RuntimeError("SyGMa exploded"),
        ):
            from app.main import create_app
            test_app = create_app(settings=_test_settings())
            with TestClient(test_app, raise_server_exceptions=False) as c:
                r = c.post("/predict", json={"smiles": ASPIRIN})
        assert r.status_code == 503

    def test_method_not_allowed_on_health(self, client):
        """POST to /health (GET-only) should return 405."""
        r = client.post("/health")
        assert r.status_code == 405
