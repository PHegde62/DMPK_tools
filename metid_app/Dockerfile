# ==============================================================================
#  MetID — Multi-Service Dockerfile
#  Runs BOTH FastAPI (port 8000) + Streamlit (port 8501) in one container.
#
#  Stages
#  ──────
#  1. conda-builder  — resolves the full conda env (cached layer)
#  2. runtime        — slim Debian + RDKit OS libs + supervisord process manager
#
#  Process management
#  ──────────────────
#  supervisord launches and supervises both services.
#  A lightweight nginx reverse-proxy is NOT used here — Hugging Face Spaces
#  and Render both expose a single public port.  The ENTRYPOINT script
#  selects which service to expose based on the EXPOSE_PORT env var:
#
#    EXPOSE_PORT=8000  →  public traffic → FastAPI  (default for Render API)
#    EXPOSE_PORT=8501  →  public traffic → Streamlit (default for HF Spaces)
#
#  Both processes always start; only the port binding differs.
# ==============================================================================

# ── Stage 1: conda-builder ────────────────────────────────────────────────────
FROM condaforge/miniforge3:24.3.0-0 AS conda-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_ENV_NAME=metid \
    CONDA_PREFIX=/opt/conda/envs/metid

COPY environment.yml /tmp/environment.yml

RUN conda env create \
        --name ${CONDA_ENV_NAME} \
        --file /tmp/environment.yml \
    && conda clean --all --yes \
    && find ${CONDA_PREFIX} -name "*.pyc"        -delete \
    && find ${CONDA_PREFIX} -name "__pycache__"  -type d -exec rm -rf {} + 2>/dev/null || true \
    && find ${CONDA_PREFIX} -name "*.a"          -delete 2>/dev/null || true


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.title="MetID — Metabolite ID & Soft Spot Analysis" \
      org.opencontainers.image.description="FastAPI + Streamlit consensus engine" \
      org.opencontainers.image.source="https://github.com/your-org/metid-app"

# ── OS dependencies ───────────────────────────────────────────────────────────
# RDKit rendering: libXrender, libXext, libGL, libgomp, libcairo, libfreetype
# Process manager: supervisor
# Health probes:   curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 \
        libxext6 \
        libgl1 \
        libgomp1 \
        libfreetype6 \
        libfontconfig1 \
        libcairo2 \
        supervisor \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN groupadd --gid 1001 appuser \
    && useradd  --uid 1001 --gid 1001 --no-create-home appuser

# ── Conda env ─────────────────────────────────────────────────────────────────
ENV CONDA_PREFIX=/opt/conda/envs/metid
COPY --from=conda-builder ${CONDA_PREFIX} ${CONDA_PREFIX}

ENV PATH="${CONDA_PREFIX}/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RDKIT_BACKEND=Agg \
    MPLBACKEND=Agg

# ── Application code ──────────────────────────────────────────────────────────
WORKDIR /app
COPY --chown=appuser:appuser . /app

# ── Streamlit config (written before switching to non-root) ──────────────────
RUN mkdir -p /home/appuser/.streamlit \
    && chown -R appuser:appuser /home/appuser
COPY --chown=appuser:appuser scripts/streamlit_config.toml /home/appuser/.streamlit/config.toml

# ── supervisord config ────────────────────────────────────────────────────────
COPY --chown=root:root scripts/supervisord.conf /etc/supervisor/conf.d/metid.conf

# ── Supervisor log dirs ───────────────────────────────────────────────────────
RUN mkdir -p /var/log/supervisor \
    && chown -R appuser:appuser /var/log/supervisor

# ── Runtime environment ───────────────────────────────────────────────────────
ENV APP_ENV=production \
    LOG_LEVEL=info \
    WORKERS=2 \
    API_PORT=8000 \
    STREAMLIT_PORT=8501 \
    METID_API_URL=http://localhost:8000 \
    CORS_ALLOW_ALL=true

# Expose both ports; cloud platforms map the one they need
EXPOSE 8000 8501

# Health-check targets the FastAPI liveness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# supervisord runs as root so it can write PID files; workers drop to appuser
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/metid.conf"]
