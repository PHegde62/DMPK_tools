"""
app/main.py
================================================================================
MetID FastAPI Application
--------------------------------------------------------------------------------
This module is intentionally self-contained: schemas, endpoints, middleware,
and the application factory all live here so the API surface is auditable in one
read-through.  Larger projects can split schemas into app/schemas/ and endpoints
into app/api/v1/; the internal structure is unchanged.

Public surface
--------------
  POST /predict
      Accepts a SMILES string plus tuning parameters.
      Validates the SMILES structurally (not just lexically), calls the
      metabolism engine, and returns a fully-typed JSON response.

  POST /render-soft-spots
      Re-runs validation + soft-spot analysis (or accepts pre-computed indices),
      then renders the parent molecule as an inline SVG with labile atoms
      highlighted in a coral-red palette.  Returns image/svg+xml.

  GET  /health   -- liveness probe (Cloud Run / App Runner / K8s)
  GET  /ready    -- readiness probe (waits for RDKit warm-up)
  GET  /docs     -- Swagger UI (disabled in production)
  GET  /redoc    -- ReDoc   (disabled in production)

CORS policy
-----------
  CORS_ORIGINS   -- comma-separated list of allowed origins
  CORS_ALLOW_ALL -- set to "true" to allow * (local dev only)
"""

from __future__ import annotations

import logging
import textwrap
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator, Dict, List, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

# RDKit is imported lazily inside functions; the lifespan hook forces the first
# import at startup so cold-request latency stays predictable.


# ==============================================================================
# Pydantic schemas
# ==============================================================================

# -- Shared SMILES validator ---------------------------------------------------

def _validate_smiles_string(v: str) -> str:
    """
    Two-layer SMILES validation used as a Pydantic field_validator.

    Layer 1 -- syntactic: reject empty / whitespace-only strings before
               touching RDKit (cheap, no C++ call).
    Layer 2 -- structural: pass through RDKit's parser + sanitisation.
               ``Chem.MolFromSmiles`` returns None for any string that fails
               valence rules, ring closure, or atom notation -- this catches
               things like "CCC(C)(C)(C)C" (pentavalent carbon) that are
               lexically valid but chemically impossible.

    We do *not* normalise or canonicalise here; that is the engine's job.
    The validator only gates clearly invalid input at the HTTP boundary.
    """
    from rdkit import Chem

    stripped = v.strip()
    if not stripped:
        raise ValueError("SMILES must be a non-empty string.")

    mol = Chem.MolFromSmiles(stripped)
    if mol is None:
        raise ValueError(
            f"'{stripped}' is not a valid SMILES string. "
            "RDKit could not parse it -- check atom symbols, valences, and "
            "ring-closure notation."
        )
    if mol.GetNumHeavyAtoms() == 0:
        raise ValueError("SMILES produced a molecule with no heavy atoms.")

    return stripped


# -- /predict request / response ----------------------------------------------

class PredictRequest(BaseModel):
    """
    Request body for ``POST /predict``.

    Fields
    ------
    smiles
        SMILES string of the parent molecule.  Must be non-empty and parseable
        by RDKit.  Salts and non-canonical notation are accepted; the engine
        normalises them internally.

    phase1_cycles / phase2_cycles
        Number of SyGMa transformation cycles per phase.  1 / 1 is the
        clinically calibrated default from the SyGMa publication.  Increasing
        these explores deeper metabolic trees at the cost of combinatorial
        explosion -- values > 2 are rarely useful in practice.

    max_metabolites
        Hard cap on the number of metabolites returned.  The engine may return
        fewer if the SyGMa tree is smaller than this limit.

    top_soft_spots
        How many labile atom sites to annotate.  3 is the recommended default
        for dashboard display; increase to 5-10 for detailed reports.

    include_svg
        When True, each metabolite in the response includes a pre-rendered SVG
        string.  This inflates the response size (~5-15 KB per metabolite) so
        it is off by default.  Use the dedicated /render-soft-spots endpoint
        for the highlighted parent depiction.
    """

    smiles: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        examples=["CC(=O)Oc1ccccc1C(=O)O"],
        description="Parent molecule SMILES (salts and non-canonical forms accepted).",
    )
    phase1_cycles: int = Field(
        default=1, ge=1, le=3,
        description="SyGMa Phase I iteration depth (1-3).",
    )
    phase2_cycles: int = Field(
        default=1, ge=1, le=3,
        description="SyGMa Phase II iteration depth (1-3).",
    )
    max_metabolites: int = Field(
        default=50, ge=1, le=500,
        description="Maximum number of metabolites to return.",
    )
    top_soft_spots: int = Field(
        default=3, ge=1, le=10,
        description="Number of soft-spot atoms to annotate.",
    )
    include_svg: bool = Field(
        default=False,
        description="Embed per-metabolite SVG strings in the response.",
    )

    @field_validator("smiles", mode="before")
    @classmethod
    def smiles_must_be_valid(cls, v: str) -> str:
        return _validate_smiles_string(v)


class SoftSpotOut(BaseModel):
    """Output representation of a single soft-spot atom."""
    atom_index:   int   = Field(description="0-based RDKit atom index.")
    atom_symbol:  str   = Field(description="Element symbol (e.g. 'C', 'N', 'S').")
    rule_name:    str   = Field(description="Matched metabolic lability rule.")
    score:        float = Field(description="Heuristic vulnerability score [0, 1].")
    smarts_match: str   = Field(description="SMARTS pattern that triggered this annotation.")


class MetaboliteOut(BaseModel):
    """Output representation of a single predicted metabolite."""
    smiles:           str
    probability:      float = Field(description="SyGMa cumulative probability [0, 1].")
    phase:            int   = Field(description="Metabolic phase (1 or 2).")
    reaction_name:    str   = Field(description="SyGMa reaction rule label.")
    molecular_weight: Optional[float] = None
    molecular_formula: Optional[str]  = None
    svg: Optional[str] = Field(
        default=None,
        description="Inline SVG (only when include_svg=True).",
    )


class ParentMoleculeOut(BaseModel):
    """Physico-chemical descriptors of the normalised parent molecule."""
    input_smiles:       str
    canonical_smiles:   str
    inchi:              str
    inchikey:           str
    molecular_formula:  str
    molecular_weight:   float
    exact_mass:         float
    num_heavy_atoms:    int
    num_rotatable_bonds: int
    num_hbd:            int
    num_hba:            int
    tpsa:               float
    logp:               float
    num_rings:          int
    num_aromatic_rings: int


class PredictResponse(BaseModel):
    """
    Full prediction response returned by ``POST /predict``.

    The ``warnings`` list carries non-fatal diagnostic messages from the engine
    (e.g. salt stripping, SyGMa unavailability).  Clients should surface these
    to the user rather than silently discarding them.
    """
    engine_version:    str
    elapsed_s:         float
    warnings:          List[str]
    parent:            ParentMoleculeOut
    metabolites:       List[MetaboliteOut]
    soft_spots:        List[SoftSpotOut]
    metabolites_total: int
    phase1_count:      int
    phase2_count:      int
    soft_spots_total:  int


# -- /render-soft-spots request -----------------------------------------------

class RenderSoftSpotsRequest(BaseModel):
    """
    Request body for ``POST /render-soft-spots``.

    Two rendering modes
    -------------------
    Mode A -- full engine run (default):
        Provide only ``smiles``.  The endpoint re-runs soft-spot analysis
        and renders the result.

    Mode B -- pre-computed indices:
        Provide ``smiles`` + ``highlight_indices``.  The engine is *not*
        called; the supplied atom indices are highlighted directly.  Use
        this when you already have a ``/predict`` response and want a
        rendering without paying the engine cost twice.

    Rendering parameters
    --------------------
    ``width`` / ``height``  -- canvas size in pixels (default 600 x 400).
    ``top_soft_spots``      -- how many atoms to highlight (Mode A only).
    ``highlight_color``     -- RGBA tuple for the highlight fill; defaults
                               to coral red (1.0, 0.35, 0.35, 0.6).
    ``show_atom_indices``   -- overlay atom index labels (debugging aid).
    ``show_scores``         -- overlay score values next to each highlighted atom.
    """

    smiles: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        examples=["CC(=O)Oc1ccccc1C(=O)O"],
    )
    highlight_indices: Optional[List[int]] = Field(
        default=None,
        description=(
            "Pre-computed atom indices to highlight (Mode B). "
            "If omitted the engine runs soft-spot analysis (Mode A)."
        ),
    )
    top_soft_spots: int = Field(
        default=3, ge=1, le=10,
        description="Number of soft spots to highlight (Mode A only).",
    )
    width:  int = Field(default=600, ge=200, le=2000)
    height: int = Field(default=400, ge=150, le=2000)
    highlight_color: List[float] = Field(
        default=[1.0, 0.35, 0.35, 0.6],
        min_length=3,
        max_length=4,
        description="RGBA tuple, each value in [0, 1]. Alpha optional (defaults to 0.6).",
    )
    show_atom_indices: bool = Field(
        default=False,
        description="Overlay RDKit atom indices on the SVG (debugging aid).",
    )
    show_scores: bool = Field(
        default=True,
        description="Annotate each highlighted atom with its vulnerability score.",
    )

    @field_validator("smiles", mode="before")
    @classmethod
    def smiles_must_be_valid(cls, v: str) -> str:
        return _validate_smiles_string(v)

    @field_validator("highlight_color", mode="before")
    @classmethod
    def validate_color(cls, v) -> List[float]:
        vals = list(v) if not isinstance(v, list) else v
        if len(vals) not in (3, 4):
            raise ValueError("highlight_color must be a list of 3 or 4 floats.")
        for c in vals:
            if not (0.0 <= float(c) <= 1.0):
                raise ValueError(
                    f"Each channel in highlight_color must be in [0, 1]; got {c}."
                )
        if len(vals) == 3:
            vals.append(0.6)
        return [float(c) for c in vals]

    @field_validator("highlight_indices", mode="before")
    @classmethod
    def validate_indices(cls, v) -> Optional[List[int]]:
        if v is None:
            return None
        indices = [int(i) for i in v]
        if any(i < 0 for i in indices):
            raise ValueError("All highlight_indices must be non-negative integers.")
        return indices

    @model_validator(mode="after")
    def check_indices_within_molecule(self) -> "RenderSoftSpotsRequest":
        """
        If pre-computed indices are supplied, verify they are within bounds
        for the parsed molecule.  Must run after all field validators.
        """
        if self.highlight_indices is None:
            return self
        from rdkit import Chem
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            return self   # already caught by the smiles validator
        n_atoms = mol.GetNumAtoms()
        oob = [i for i in self.highlight_indices if i >= n_atoms]
        if oob:
            raise ValueError(
                f"highlight_indices {oob} are out of range for a molecule "
                f"with {n_atoms} atoms (0-based indexing)."
            )
        return self


# -- Error response schema ----------------------------------------------------

class ErrorDetail(BaseModel):
    """Structured error body returned by all 4xx / 5xx responses."""
    status_code: int
    error:       str
    detail:      str
    path:        Optional[str] = None


# ==============================================================================
# SVG renderer
# ==============================================================================

def _render_soft_spot_svg(
    smiles: str,
    highlight_indices: List[int],
    highlight_scores: Optional[Dict[int, float]],
    width: int,
    height: int,
    color_rgba: List[float],
    show_atom_indices: bool,
    show_scores: bool,
) -> str:
    """
    Render the parent molecule as an SVG with soft-spot atoms highlighted.

    Implementation notes
    --------------------
    We use ``rdMolDraw2D.MolDraw2DSVG`` rather than the higher-level
    ``Draw.MolToImage`` because it gives us precise control over:

      - Per-atom highlight colours (``highlightAtomColors``)
      - Per-bond highlight colours (bonds between highlighted atoms are also
        coloured, making metabolic hotspot *regions* visually obvious)
      - Font sizes, padding, and atom index overlay
      - Clean SVG output with no filesystem I/O (purely in-memory)

    Highlight bonds
    ---------------
    Any bond where *both* endpoint atoms are in the highlight set is tinted
    with a slightly lighter version of the highlight colour to show metabolic
    regions rather than isolated atoms -- especially useful for aromatic rings
    where multiple carbons are flagged.

    Score annotations
    -----------------
    When ``show_scores=True`` we inject score labels as SVG <text> elements
    positioned above each highlighted atom's 2-D coordinate.  We post-process
    the SVG string rather than using an XML parser to avoid an lxml dependency.

    Parameters
    ----------
    smiles            : canonical (or raw) SMILES of the parent molecule.
    highlight_indices : atom indices to colour.
    highlight_scores  : mapping of atom_index -> score for label overlay.
    width / height    : SVG canvas size in pixels.
    color_rgba        : [R, G, B, A] each in [0, 1].
    show_atom_indices : toggle RDKit's built-in index overlay.
    show_scores       : toggle score label injection.

    Returns
    -------
    SVG string (UTF-8).

    Raises
    ------
    ValueError  if the SMILES is unparseable.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    # -- Parse + 2-D layout ---------------------------------------------------
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot render: invalid SMILES '{smiles}'")

    # CoordGen produces cleaner layouts for fused rings vs the default
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)

    # -- Colour maps ----------------------------------------------------------
    r, g, b, a = color_rgba

    # Primary highlight: the exact colour supplied by the caller
    atom_color = (r, g, b)

    # Bond highlight: a lighter tint of the same hue
    bond_color = (
        min(1.0, r + (1.0 - r) * 0.35),
        min(1.0, g + (1.0 - g) * 0.35),
        min(1.0, b + (1.0 - b) * 0.35),
    )

    highlight_set = set(highlight_indices)
    atom_color_map: Dict[int, tuple] = {idx: atom_color for idx in highlight_set}

    # Identify bonds where both endpoints are highlighted
    highlighted_bonds: List[int] = []
    bond_color_map: Dict[int, tuple] = {}
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i in highlight_set and j in highlight_set:
            highlighted_bonds.append(bond.GetIdx())
            bond_color_map[bond.GetIdx()] = bond_color

    # -- Drawer configuration -------------------------------------------------
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()

    opts.addStereoAnnotation        = True
    opts.addAtomIndices             = show_atom_indices
    opts.atomHighlightsAreCircles   = False   # filled wedge style
    opts.fillHighlights             = True
    opts.highlightRadius            = 0.35    # Angstrom-scale radius
    opts.padding                    = 0.15    # fraction of canvas left as margin

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_set),
        highlightAtomColors=atom_color_map,
        highlightBonds=highlighted_bonds,
        highlightBondColors=bond_color_map,
    )
    drawer.FinishDrawing()
    svg: str = drawer.GetDrawingText()

    # -- Score label injection ------------------------------------------------
    if show_scores and highlight_scores:
        svg = _inject_score_labels(
            svg=svg,
            mol=mol,
            drawer=drawer,
            highlight_scores=highlight_scores,
            color_rgba=color_rgba,
        )

    return svg


def _inject_score_labels(
    svg: str,
    mol,
    drawer,
    highlight_scores: Dict[int, float],
    color_rgba: List[float],
) -> str:
    """
    Post-process an RDKit SVG to overlay score labels above highlighted atoms.

    Strategy
    --------
    RDKit's ``GetDrawCoords`` returns the 2-D canvas coordinates (pixels) for
    each atom.  We build <text> SVG elements positioned slightly above each
    atom centre and splice them before the closing </svg> tag.

    Each label has a semi-transparent white backing <rect> for legibility
    against dark bond lines.  The label colour is a darkened variant of the
    highlight hue.
    """
    r, g, b, _ = color_rgba
    label_r = int(r * 180)
    label_g = int(g * 130)
    label_b = int(b * 130)
    label_color = f"rgb({label_r},{label_g},{label_b})"

    label_elements: List[str] = []

    for atom_idx, score in sorted(highlight_scores.items()):
        try:
            pt = drawer.GetDrawCoords(atom_idx)
            cx, cy = pt.x, pt.y
        except Exception:
            continue  # coords unavailable -- skip this label silently

        # Position label 18 px above atom centre
        lx = cx
        ly = cy - 18

        # White backing rectangle
        rect_w, rect_h = 34, 14
        rect_x = lx - rect_w / 2
        rect_y = ly - rect_h + 2

        label_elements.append(
            f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" '
            f'width="{rect_w}" height="{rect_h}" '
            f'rx="3" ry="3" fill="white" fill-opacity="0.80" stroke="none"/>'
        )
        label_elements.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" '
            f'text-anchor="middle" font-size="10" font-weight="bold" '
            f'font-family="monospace" fill="{label_color}">'
            f'{score:.2f}</text>'
        )

    if not label_elements:
        return svg

    injected_block = "\n".join(label_elements)
    return svg.replace("</svg>", f"\n{injected_block}\n</svg>")


# ==============================================================================
# Lifespan (startup / shutdown)
# ==============================================================================

# Module-level flag set to True once RDKit has been warm-started.
# The /ready probe checks this before returning 200.
_rdkit_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Startup tasks
    -------------
    1. Configure structured logging (structlog -> JSON in prod, console in dev).
    2. Force-import RDKit to pay the C-extension load cost once at startup rather
       than on the first real request.
    3. Warn if SyGMa is absent so the log reflects the deployment state.

    Shutdown tasks
    --------------
    1. Dispose of the SQLAlchemy async engine pool.
    2. Emit a structured shutdown log.
    """
    global _rdkit_ready

    cfg = get_settings()
    configure_logging()
    logger.info("startup", env=cfg.APP_ENV, version=cfg.APP_VERSION)

    # -- Warm-up RDKit --------------------------------------------------------
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D  # noqa: F401

        # Touch a molecule to initialise the descriptor tables
        _ = Chem.MolFromSmiles("C")
        rdDepictor.Compute2DCoords(Chem.MolFromSmiles("C"))
        _rdkit_ready = True
        logger.info("rdkit_warm_up_complete")
    except Exception as exc:
        logger.error("rdkit_warm_up_failed", error=str(exc))

    # -- Warn if SyGMa is absent ----------------------------------------------
    try:
        import sygma  # type: ignore  # noqa: F401
        logger.info("sygma_available")
    except ImportError:
        logger.warning(
            "sygma_not_installed",
            hint="Install via 'pip install sygma'. Metabolite prediction will return [].",
        )

    yield   # <-- application runs here

    # -- Shutdown -------------------------------------------------------------
    try:
        from app.db.session import engine as db_engine
        await db_engine.dispose()
    except Exception:
        pass

    logger.info("shutdown_complete")


# ==============================================================================
# Application factory
# ==============================================================================

def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    FastAPI application factory.

    Accepting ``settings`` as a parameter makes the factory fully testable:
    tests can inject a ``Settings`` instance with overrides (e.g. a permissive
    CORS policy) without touching environment variables.

    Parameters
    ----------
    settings : Settings instance.  Defaults to the cached global singleton.

    Returns
    -------
    Configured ``FastAPI`` application.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="MetID API",
        description=textwrap.dedent("""
            **Metabolite Identification (MetID) & Soft Spot Analysis API**

            Predict Phase I / Phase II metabolites of small molecules using
            SyGMa rule-based metabolism and identify metabolically labile atoms
            (soft spots) via SMARTS substructure rules.
        """).strip(),
        version=settings.APP_VERSION,
        docs_url="/docs"  if settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if settings.APP_ENV != "production" else None,
        openapi_url="/openapi.json" if settings.APP_ENV != "production" else None,
        lifespan=lifespan,
    )

    # -- CORS -----------------------------------------------------------------
    # Policy:
    #   CORS_ALLOW_ALL=true  ->  allow * (dev only; credentials disabled)
    #   Otherwise            ->  only origins in CORS_ORIGINS are allowed
    #
    # Browsers reject credentialed requests when allow_origins=["*"] so we
    # disable allow_credentials in that case.
    if settings.CORS_ALLOW_ALL:
        cors_origins = ["*"]
        cors_creds   = False
        cors_headers = ["*"]
    else:
        cors_origins = [str(o) for o in settings.CORS_ORIGINS]
        cors_creds   = True
        cors_headers = [
            "Authorization",
            "Content-Type",
            "Accept",
            "X-Request-ID",
            "X-Correlation-ID",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_creds,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=cors_headers,
        max_age=600,   # cache preflight responses for 10 minutes
    )

    # -- Gzip -----------------------------------------------------------------
    # Metabolite JSON responses can be 50-200 KB; gzip gives 5-10x compression.
    app.add_middleware(GZipMiddleware, minimum_size=512)

    # -- Global exception handler ---------------------------------------------
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "unhandled_exception", path=request.url.path, error=str(exc)
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorDetail(
                status_code=500,
                error="InternalServerError",
                detail="An unexpected error occurred. Check server logs for details.",
                path=str(request.url.path),
            ).model_dump(),
        )

    # -- Register routes ------------------------------------------------------
    _register_routes(app, settings)

    return app


# ==============================================================================
# Route definitions
# ==============================================================================

def _register_routes(app: FastAPI, settings: Settings) -> None:
    """
    Attach all route handlers to ``app``.

    Defined as a standalone function (rather than module-level decorators)
    so the factory registers routes *after* all middleware is added, which
    is the correct FastAPI setup order.
    """

    # -- Probes ---------------------------------------------------------------

    @app.get(
        "/health",
        tags=["ops"],
        summary="Liveness probe",
        response_description="Always 200 if the process is alive.",
    )
    async def health() -> Dict:
        """
        Liveness probe.

        Cloud Run / App Runner use this to decide whether to restart the
        container.  It does *not* check downstream dependencies.
        """
        return {"status": "ok", "version": settings.APP_VERSION}

    @app.get(
        "/ready",
        tags=["ops"],
        summary="Readiness probe",
        response_description="Returns 200 once RDKit has completed warm-up.",
    )
    async def ready() -> JSONResponse:
        """
        Readiness probe.

        Returns 503 until RDKit has completed its startup warm-up so that
        load balancers do not route traffic to a partially-initialised container.
        """
        if not _rdkit_ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "RDKit initialising"},
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ready"},
        )

    # -- POST /predict --------------------------------------------------------

    @app.post(
        "/predict",
        response_model=PredictResponse,
        status_code=status.HTTP_200_OK,
        tags=["prediction"],
        summary="Predict Phase I / II metabolites and soft spots",
        responses={
            200: {"description": "Successful prediction"},
            422: {
                "description": "Invalid SMILES or request parameters",
                "model": ErrorDetail,
            },
            503: {
                "description": "SyGMa engine unavailable",
                "model": ErrorDetail,
            },
        },
    )
    async def predict_endpoint(
        body: Annotated[PredictRequest, Body(...)],
        cfg:  Settings = Depends(get_settings),
    ) -> PredictResponse:
        """
        Run the full metabolism prediction pipeline for a parent molecule.

        **Pipeline steps**

        1. The SMILES is validated structurally via RDKit (Pydantic validator).
        2. The normalisation pipeline strips salts, normalises functional groups,
           and uncharges the molecule.
        3. SyGMa runs Phase I transformations followed by Phase II conjugations.
        4. The SMARTS-based soft-spot analyser identifies the top-N labile atoms.
        5. A structured JSON response is returned.

        The ``warnings`` array carries non-fatal diagnostic messages from the
        engine (e.g. salt stripping, SyGMa not installed).  Clients should
        surface these to the user rather than silently discarding them.
        """
        from app.engine.metabolism import predict as engine_predict

        logger.info(
            "predict_request",
            smiles=body.smiles[:80],
            phase1_cycles=body.phase1_cycles,
            phase2_cycles=body.phase2_cycles,
        )

        try:
            result = engine_predict(
                smiles=body.smiles,
                phase1_cycles=body.phase1_cycles,
                phase2_cycles=body.phase2_cycles,
                max_metabolites=min(
                    body.max_metabolites, cfg.MAX_METABOLITES_RETURNED
                ),
                top_soft_spots=body.top_soft_spots,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            logger.error("engine_runtime_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Metabolism engine error: {exc}",
            ) from exc

        # -- Build response ---------------------------------------------------
        raw = result.to_dict()

        metabolites_out: List[MetaboliteOut] = []
        for m in raw["metabolites"]:
            svg_str: Optional[str] = None
            if body.include_svg:
                try:
                    svg_str = _render_soft_spot_svg(
                        smiles=m["smiles"],
                        highlight_indices=[],
                        highlight_scores=None,
                        width=300,
                        height=200,
                        color_rgba=[1.0, 0.35, 0.35, 0.6],
                        show_atom_indices=False,
                        show_scores=False,
                    )
                except Exception:
                    pass   # SVG failure must not fail the whole response
            metabolites_out.append(MetaboliteOut(**m, svg=svg_str))

        soft_spots_out = [SoftSpotOut(**s) for s in raw["soft_spots"]]
        parent_out     = ParentMoleculeOut(**raw["parent"])

        response = PredictResponse(
            engine_version=raw["engine_version"],
            elapsed_s=raw["elapsed_s"],
            warnings=raw["warnings"],
            parent=parent_out,
            metabolites=metabolites_out,
            soft_spots=soft_spots_out,
            metabolites_total=len(metabolites_out),
            phase1_count=sum(1 for m in metabolites_out if m.phase == 1),
            phase2_count=sum(1 for m in metabolites_out if m.phase == 2),
            soft_spots_total=len(soft_spots_out),
        )

        logger.info(
            "predict_response",
            canonical_smiles=raw["parent"]["canonical_smiles"],
            metabolites_total=response.metabolites_total,
            soft_spots_total=response.soft_spots_total,
            elapsed_s=response.elapsed_s,
        )

        return response

    # -- POST /render-soft-spots ----------------------------------------------

    @app.post(
        "/render-soft-spots",
        response_class=Response,
        status_code=status.HTTP_200_OK,
        tags=["rendering"],
        summary="Render parent molecule SVG with soft-spot atoms highlighted",
        responses={
            200: {
                "description": "SVG image of the molecule with highlighted soft spots.",
                "content": {"image/svg+xml": {}},
            },
            422: {
                "description": "Invalid SMILES, atom indices out of range, or bad colour.",
                "model": ErrorDetail,
            },
            500: {
                "description": "SVG rendering failed.",
                "model": ErrorDetail,
            },
        },
    )
    async def render_soft_spots(
        body: Annotated[RenderSoftSpotsRequest, Body(...)],
    ) -> Response:
        """
        Render the parent molecule as an SVG with soft-spot atoms highlighted.

        **Two modes**

        - **Mode A** *(default)*: omit ``highlight_indices``.  The soft-spot
          analysis engine runs automatically.

        - **Mode B**: supply ``highlight_indices`` from a previous ``/predict``
          response.  The engine is skipped -- no double cost.

        **Returns**

        ``image/svg+xml`` response.  The SVG can be embedded directly in an
        ``<img src="data:image/svg+xml;base64,...">`` tag or as an inline
        ``<svg>`` element in your React / Vue dashboard.

        **Highlight colour**

        The default coral-red ``[1.0, 0.35, 0.35, 0.6]`` was chosen for high
        contrast on white backgrounds while remaining distinguishable from the
        standard RDKit bond/atom colours.  Supply a custom ``highlight_color``
        RGBA tuple to match your dashboard theme.
        """
        from app.engine.metabolism import (
            _find_soft_spots,
            _validate_and_normalise,
        )

        # -- Resolve highlight indices ----------------------------------------
        highlight_scores: Dict[int, float] = {}

        if body.highlight_indices is not None:
            # Mode B: use supplied indices directly
            indices = body.highlight_indices
            # Scores unknown; fill with 1.0 so labels show "1.00"
            highlight_scores = {i: 1.0 for i in indices}
        else:
            # Mode A: run engine soft-spot analysis
            try:
                mol, _ = _validate_and_normalise(body.smiles)
                soft_spots = _find_soft_spots(mol, top_n=body.top_soft_spots)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc

            indices = [s.atom_index for s in soft_spots]
            highlight_scores = {s.atom_index: s.score for s in soft_spots}

        if not indices:
            logger.info("render_no_soft_spots", smiles=body.smiles[:80])

        # -- Render SVG -------------------------------------------------------
        try:
            svg = _render_soft_spot_svg(
                smiles=body.smiles,
                highlight_indices=indices,
                highlight_scores=highlight_scores if body.show_scores else None,
                width=body.width,
                height=body.height,
                color_rgba=body.highlight_color,
                show_atom_indices=body.show_atom_indices,
                show_scores=body.show_scores,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception(
                "svg_render_error", smiles=body.smiles[:80], error=str(exc)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SVG rendering failed. See server logs for details.",
            ) from exc

        logger.info(
            "render_soft_spots_response",
            smiles=body.smiles[:80],
            highlighted_atoms=indices,
            svg_bytes=len(svg),
        )

        # Cache the SVG client-side for 60 s (molecules are deterministic)
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=60",
                # Convenience header for frontend to read highlighted atoms
                # without parsing the SVG.
                "X-Highlighted-Atoms": ",".join(str(i) for i in indices),
                "X-Soft-Spot-Count": str(len(indices)),
            },
        )


# ==============================================================================
# Application instance
# ==============================================================================

#: Module-level singleton used by uvicorn and tests.
#: ``uvicorn app.main:app --reload``
app: FastAPI = create_app()
