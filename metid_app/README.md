# ⚗️ MetID — Metabolite Intelligence Platform

[![CI / CD](https://github.com/your-org/metid-app/actions/workflows/deploy.yml/badge.svg)](https://github.com/your-org/metid-app/actions/workflows/deploy.yml)
[![Deploy to HF Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-to-spaces-sm.svg)](https://huggingface.co/spaces/your-username/metid-app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **MetID** is a production-grade drug metabolism prediction platform combining rule-based SyGMa
> reaction matrices with a deep-learning sequence translation emulator (MetaTrans / Meta-Predictor
> architecture) into an **Ensemble Consensus Engine** that tags metabolites by confidence tier.

---

## Contents

- [Architecture](#architecture)
- [Consensus Engine](#consensus-engine)
- [Repository Structure](#repository-structure)
- [Local Setup (Conda)](#local-setup)
- [Running the Full Stack](#running-the-full-stack)
- [Docker](#docker)
- [Deploy to Cloud](#deploy-to-cloud)
- [API Reference](#api-reference)
- [Integrating Real DL Weights](#integrating-real-dl-weights)
- [Contributing](#contributing)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MetID Ensemble Architecture                        │
├──────────────────────────┬──────────────────────────────────────────────┤
│  Pipeline A (Rule-Based) │  Pipeline B (Deep Learning)                  │
│  ─────────────────────── │  ─────────────────────────────────────────── │
│  SyGMa SMIRKS matrices   │  DeepLearningPredictor class                 │
│  Phase I (CYP oxidation, │  • MetaTrans / Meta-Predictor interface       │
│    reduction, hydrolysis)│  • Atom-level SMILES tokeniser               │
│  Phase II (glucuronid.,  │  • Transformer beam-search hook              │
│    sulfation, acetylation│  • Per-atom attention weight extraction       │
│  → PredictedMetabolite[] │  → (predictions[], attention{}, warnings[])  │
├──────────────────────────┴──────────────────────────────────────────────┤
│                        Consensus Engine                                 │
│  RDKit CanonSmiles alignment → confidence-tier tagging                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Both pipelines agree   →  High Confidence (Consensus Verified)   │ │
│  │  SyGMa only             →  Moderate Confidence (Rule-Only)        │ │
│  │  DL only                →  Moderate Confidence (DL-Only)          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│  Unified Vulnerability Index:                                           │
│    VI = (0.55 × rule_score + 0.45 × dl_attention_risk) × 100 %         │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓ MetabolismResult
              FastAPI (/predict, /render-soft-spots)
                            ↓
              Streamlit Dashboard (3-panel layout)
```

### Inspired by / compatible with

| Repository | Role |
|---|---|
| [3D-e-Chem/sygma](https://github.com/3D-e-Chem/sygma) | Phase I/II SMIRKS rule engine |
| [KavrakiLab/MetaTrans](https://github.com/KavrakiLab/MetaTrans) | Seq2seq SMILES translation architecture |
| [zhukeyun/Meta-Predictor](https://github.com/zhukeyun/Meta-Predictor) | Attention-based Site-of-Metabolism |
| [rdkit/rdkit](https://github.com/rdkit/rdkit) | Cheminformatics, SVG rendering, canonicalisation |

---

## Consensus Engine

The `ConsensusEngine` class cross-validates both pipelines using
`Chem.MolToSmiles(mol, canonical=True)` for exact structural alignment:

```python
from app.engine.metabolism import predict, DeepLearningPredictor

# Default: ensemble mode with emulator
result = predict("CC(=O)Oc1ccccc1C(=O)O")

# With real MetaTrans weights (when available)
predictor = DeepLearningPredictor.from_checkpoint("weights/metatrans.pt")
result = predict("CC(=O)Oc1ccccc1C(=O)O", dl_predictor=predictor)

# Rule-only mode (v1 backward compat)
result = predict("CC(=O)Oc1ccccc1C(=O)O", enable_dl=False)

print(result.parent.molecular_formula)   # C9H8O4
for m in result.consensus_metabolites:
    print(m.smiles, m.confidence_tier)

for s in result.soft_spots:
    print(f"Atom {s.atom_index}: VI={s.vulnerability_index:.1f}%")
```

---

## Repository Structure

```
metid-app/
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD: quality → test → docker → deploy
│
├── app/
│   ├── engine/
│   │   └── metabolism.py       # ★ Ensemble Consensus Engine (v2)
│   ├── core/
│   │   ├── config.py           # Pydantic-settings (env vars)
│   │   └── logging.py          # structlog JSON logging
│   ├── db/
│   │   └── session.py          # Async SQLAlchemy engine
│   ├── main.py                 # FastAPI app factory + endpoints
│   └── frontend.py             # ★ Streamlit premium dashboard
│
├── tests/
│   ├── unit/
│   │   ├── test_ensemble_engine.py   # 100+ engine tests
│   │   ├── test_main.py              # 56 API layer tests
│   │   └── test_frontend.py          # 43 frontend helper tests
│   └── integration/
│       └── test_metabolites_endpoint.py
│
├── scripts/
│   ├── supervisord.conf        # Process manager (FastAPI + Streamlit)
│   ├── streamlit_config.toml   # Streamlit theme + server settings
│   └── run_dashboard.sh        # Local dev launcher
│
├── Dockerfile                  # Multi-stage: conda-builder → runtime
├── docker-compose.yml          # Local dev: API + Streamlit + Postgres + Redis
├── environment.yml             # Conda-forge dependency spec
├── .env.example                # Environment variable template
└── README.md
```

---

## Local Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
  [Miniforge](https://github.com/conda-forge/miniforge) ≥ 23.x
- Git
- Docker Desktop (optional, for container workflow)

### 1. Clone

```bash
git clone https://github.com/your-org/metid-app.git
cd metid-app
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate metid
```

> **Apple Silicon (M1/M2):** RDKit and SyGMa are distributed as
> `osx-arm64` conda-forge packages; no Rosetta needed.

### 3. Copy and edit environment variables

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL, REDIS_URL, CORS_ORIGINS as needed
```

### 4. Start the FastAPI backend

```bash
uvicorn app.main:app --reload --port 8000
# → http://localhost:8000/docs
```

### 5. Start the Streamlit frontend (new terminal)

```bash
METID_API_URL=http://localhost:8000 \
streamlit run app/frontend.py --server.port 8501
# → http://localhost:8501
```

### 6. Run tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

### 7. Format & lint

```bash
black app/ tests/
ruff check app/ tests/
```

---

## Running the Full Stack (Docker Compose)

```bash
# Start API + Streamlit + PostgreSQL + Redis
docker compose up --build

# Services:
#   FastAPI   → http://localhost:8000
#   Streamlit → http://localhost:8501
#   Postgres  → localhost:5432
#   Redis     → localhost:6379
```

---

## Docker

Build and run the single-container image (both services via supervisord):

```bash
# Build
docker build -t metid-app:latest .

# Run (both ports)
docker run -p 8000:8000 -p 8501:8501 \
  -e APP_ENV=production \
  -e CORS_ALLOW_ALL=true \
  metid-app:latest
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `API_PORT` | `8000` | FastAPI uvicorn port |
| `STREAMLIT_PORT` | `8501` | Streamlit server port |
| `WORKERS` | `2` | uvicorn worker count |
| `LOG_LEVEL` | `info` | logging verbosity |
| `CORS_ORIGINS` | `""` | comma-separated allowed origins |
| `CORS_ALLOW_ALL` | `false` | set `true` for dev wildcard CORS |
| `METID_API_URL` | `http://localhost:8000` | Streamlit → API URL |

---

## Deploy to Cloud

### Hugging Face Spaces (Recommended — free GPU tier available)

1. Fork this repository
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Docker**
   - Hardware: CPU Basic (or upgrade for GPU inference)
3. Add repository secrets in **GitHub → Settings → Secrets and variables → Actions**:
   ```
   HF_TOKEN          your-huggingface-write-token
   HF_SPACE_ID       your-username/metid-app
   ```
4. Push to `main` — the `deploy-hf` job runs automatically

HF Spaces exposes **port 7860** by default.  Add to your `supervisord.conf`:
```
STREAMLIT_PORT=7860
```
or set the `APP_PORT` Space hardware variable.

---

### Render

1. Fork this repository
2. Create a new **Web Service** on [render.com](https://render.com)
   - Environment: **Docker**
   - Port: `8000` (FastAPI) or `8501` (Streamlit)
3. Add repository secret:
   ```
   RENDER_DEPLOY_HOOK    <your-render-deploy-hook-url>
   ```
4. Push to `main` — the `deploy-render` job triggers Render automatically

---

### GitHub Actions secrets summary

| Secret | Where to find it |
|---|---|
| `HF_TOKEN` | HuggingFace → Settings → Access Tokens → write |
| `HF_SPACE_ID` | `your-username/your-space-name` |
| `RENDER_DEPLOY_HOOK` | Render → Service → Settings → Deploy Hook |

---

## API Reference

### `POST /predict`

```json
{
  "smiles":          "CC(=O)Oc1ccccc1C(=O)O",
  "phase1_cycles":   1,
  "phase2_cycles":   1,
  "max_metabolites": 50,
  "top_soft_spots":  3,
  "include_svg":     false
}
```

Response includes:
- `parent` — physico-chemical descriptors
- `metabolites[]` — with `confidence_tier`, `dl_score`, `sources`
- `soft_spots[]` — with `rule_score`, `dl_attention_risk`, `vulnerability_index`
- `pipeline_stats` — per-pipeline counts + consensus stats

### `POST /render-soft-spots`

```json
{
  "smiles":            "CC(=O)Oc1ccccc1C(=O)O",
  "highlight_indices": [7, 1, 4],
  "width":             560,
  "height":            380,
  "highlight_color":   [0.98, 0.25, 0.25, 0.70],
  "show_scores":       true
}
```

Returns `image/svg+xml` with atom highlights.

### `GET /health` · `GET /ready`

Liveness and readiness probes for Cloud Run / App Runner / K8s.

Interactive docs available at `/docs` (non-production only).

---

## Integrating Real DL Weights

The `DeepLearningPredictor` class is a drop-in interface designed for
exact compatibility with MetaTrans and Meta-Predictor checkpoints.

### Step 1 — Install PyTorch + OpenNMT

```bash
# Add to environment.yml pip section:
# - torch>=2.2
# - opennmt-py>=3.4
conda env update -f environment.yml
```

### Step 2 — Override the inference hooks

```python
# app/engine/dl_real.py
import torch
import onmt
from app.engine.metabolism import DeepLearningPredictor

class MetaTransPredictor(DeepLearningPredictor):

    def __init__(self, checkpoint_path: str, device: str = "cpu", **kw):
        super().__init__(**kw)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Build model from checkpoint opt + fields
        self._model = onmt.model_builder.build_base_model(
            checkpoint["opt"], checkpoint["vocab"], gpu=(device != "cpu")
        )
        self._model.load_state_dict(checkpoint["model"])
        self._model.eval()
        self._device = device
        self._model_loaded = True

    def _transformer_beam_search(self, tokens, top_k):
        # Real OpenNMT beam-search call
        with torch.no_grad():
            results = onmt.translate.translate(
                src=" ".join(tokens),
                model=self._model,
                beam_size=top_k,
            )
        return [(r.pred_sents[0], float(r.scores[0])) for r in results]

    def _extract_attention_weights(self, mol, tokens):
        # Extract encoder attention for SOM risk
        with torch.no_grad():
            _, attentions = self._model.encoder(tokens)
        # Map attention heads to atom indices ...
        return {i: float(attentions[0][0][i].mean()) for i in range(mol.GetNumHeavyAtoms())}
```

### Step 3 — Wire into predict()

```python
from app.engine.dl_real import MetaTransPredictor
from app.engine.metabolism import predict

predictor = MetaTransPredictor.from_checkpoint(
    "weights/metatrans_pubchem.pt", device="cuda"
)
result = predict("CC(=O)Oc1ccccc1C(=O)O", dl_predictor=predictor)
```

No changes to the FastAPI layer, Streamlit frontend, or test suite required.

---

## Contributing

1. Fork → feature branch → PR against `main`
2. `black app/ tests/` and `ruff check app/ tests/` must pass (enforced by CI)
3. Add / update tests — coverage threshold: 80 %
4. Update `CHANGELOG.md` with a brief description

---

## License

[MIT](LICENSE) — © 2024 your-org.

*Soft spot scores and metabolite predictions are computational estimates
for research use only and have not been clinically validated.*
