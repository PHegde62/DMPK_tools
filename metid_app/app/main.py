"""
app/main.py — Lightweight MetID FastAPI for Render free tier.
Only depends on: fastapi, uvicorn, pydantic, rdkit, httpx, structlog
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

APP_VERSION = "2.0.0-ensemble"
_rdkit_ready = False


# ── SMILES validator ──────────────────────────────────────────────────────────
def _validate_smiles(smiles: str) -> str:
    from rdkit import Chem
    stripped = smiles.strip()
    if not stripped:
        raise ValueError("SMILES must be a non-empty string.")
    mol = Chem.MolFromSmiles(stripped)
    if mol is None:
        raise ValueError(f"Invalid SMILES: '{stripped}'")
    if mol.GetNumHeavyAtoms() == 0:
        raise ValueError("SMILES produced a molecule with no heavy atoms.")
    return stripped


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=4096)
    phase1_cycles: int = Field(default=1, ge=1, le=3)
    phase2_cycles: int = Field(default=1, ge=1, le=3)
    max_metabolites: int = Field(default=50, ge=1, le=500)
    top_soft_spots: int = Field(default=3, ge=1, le=10)
    include_svg: bool = False

    @field_validator("smiles", mode="before")
    @classmethod
    def validate_smiles(cls, v):
        return _validate_smiles(v)


class RenderRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=4096)
    highlight_indices: Optional[List[int]] = None
    top_soft_spots: int = Field(default=3, ge=1, le=10)
    width: int = Field(default=560, ge=200, le=2000)
    height: int = Field(default=380, ge=150, le=2000)
    highlight_color: List[float] = Field(default=[1.0, 0.35, 0.35, 0.65])
    show_scores: bool = True
    show_atom_indices: bool = False

    @field_validator("smiles", mode="before")
    @classmethod
    def validate_smiles(cls, v):
        return _validate_smiles(v)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _rdkit_ready
    logger.info(f"Starting MetID API v{APP_VERSION}")
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        _ = Chem.MolFromSmiles("C")
        rdDepictor.Compute2DCoords(Chem.MolFromSmiles("C"))
        _rdkit_ready = True
        logger.info("RDKit ready")
    except Exception as e:
        logger.error(f"RDKit warm-up failed: {e}")
    yield
    logger.info("Shutdown complete")


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="MetID API",
        description="Metabolite Identification & Soft Spot Analysis",
        version=APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(GZipMiddleware, minimum_size=512)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=600,
    )

    @app.get("/health", tags=["ops"])
    async def health():
        return {"status": "ok", "version": APP_VERSION}

    @app.get("/ready", tags=["ops"])
    async def ready():
        if not _rdkit_ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "RDKit initialising"},
            )
        return {"status": "ready"}

    @app.post("/predict", tags=["prediction"])
    async def predict(body: PredictRequest):
        from app.engine.metabolism import predict as engine_predict
        try:
            result = engine_predict(
                smiles=body.smiles,
                phase1_cycles=body.phase1_cycles,
                phase2_cycles=body.phase2_cycles,
                max_metabolites=body.max_metabolites,
                top_soft_spots=body.top_soft_spots,
            )
            d = result.to_dict()
            # Add convenience counts
            mets = d.get("metabolites", [])
            d["metabolites_total"] = len(mets)
            d["phase1_count"] = sum(1 for m in mets if m.get("phase") == 1)
            d["phase2_count"] = sum(1 for m in mets if m.get("phase") == 2)
            d["soft_spots_total"] = len(d.get("soft_spots", []))
            return d
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"Predict error: {e}")
            raise HTTPException(status_code=503, detail=str(e))

    @app.post("/render-soft-spots", tags=["rendering"])
    async def render_soft_spots(body: RenderRequest):
        from app.engine.metabolism import _find_soft_spots, _validate_and_normalise
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D

        try:
            mol, _ = _validate_and_normalise(body.smiles)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        if body.highlight_indices is not None:
            indices = body.highlight_indices
            scores = {i: 1.0 for i in indices}
        else:
            spots = _find_soft_spots(mol, top_n=body.top_soft_spots)
            indices = [s.atom_index for s in spots]
            scores = {s.atom_index: s.score for s in spots}

        try:
            rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(mol)

            r, g, b = body.highlight_color[0], body.highlight_color[1], body.highlight_color[2]
            atom_color = (r, g, b)
            bond_color = (
                min(1.0, r + (1.0 - r) * 0.35),
                min(1.0, g + (1.0 - g) * 0.35),
                min(1.0, b + (1.0 - b) * 0.35),
            )

            highlight_set = set(indices)
            atom_color_map = {i: atom_color for i in highlight_set}
            highlighted_bonds = []
            bond_color_map = {}
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if i in highlight_set and j in highlight_set:
                    highlighted_bonds.append(bond.GetIdx())
                    bond_color_map[bond.GetIdx()] = bond_color

            drawer = rdMolDraw2D.MolDraw2DSVG(body.width, body.height)
            opts = drawer.drawOptions()
            opts.addStereoAnnotation = True
            opts.fillHighlights = True
            opts.highlightRadius = 0.35
            opts.padding = 0.15

            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(highlight_set),
                highlightAtomColors=atom_color_map,
                highlightBonds=highlighted_bonds,
                highlightBondColors=bond_color_map,
            )
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()

            return Response(
                content=svg,
                media_type="image/svg+xml",
                headers={
                    "Cache-Control": "public, max-age=60",
                    "X-Highlighted-Atoms": ",".join(str(i) for i in indices),
                    "X-Soft-Spot-Count": str(len(indices)),
                },
            )
        except Exception as e:
            logger.error(f"Render error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()
