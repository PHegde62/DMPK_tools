"""
app/core/config.py — Centralised settings via pydantic-settings.

All values can be overridden by environment variables (case-insensitive).
Secrets (DB password, API keys) should be injected via your cloud provider's
secret manager (AWS Secrets Manager / GCP Secret Manager), NOT baked into images.
"""

from functools import lru_cache
from typing import List

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_ENV: str = "development"          # development | staging | production
    APP_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "info"
    SECRET_KEY: str = "change-me-in-production"

    # ── Server ────────────────────────────────────────────────────────────────
    PORT: int = 8080
    WORKERS: int = 2
    CORS_ORIGINS: List[AnyHttpUrl] = []

    # CORS_ALLOW_ALL: set to "true" to allow any origin (local dev only).
    # Must NEVER be "true" in production -- credentials are disabled automatically.
    CORS_ALLOW_ALL: bool = False

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./metid_dev.db"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # ── Redis / Celery ────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── Cloud storage ─────────────────────────────────────────────────────────
    STORAGE_BACKEND: str = "local"        # local | s3 | gcs
    S3_BUCKET: str = ""
    GCS_BUCKET: str = ""
    LOCAL_STORAGE_PATH: str = "/tmp/metid_results"

    # ── SyGMa / metabolite engine ─────────────────────────────────────────────
    SYGMA_PHASE1_CYCLES: int = 1
    SYGMA_PHASE2_CYCLES: int = 1
    MAX_METABOLITES_RETURNED: int = 200

    # ── ML plug-in registry ───────────────────────────────────────────────────
    # Comma-separated list of enabled predictor module paths.
    # e.g. "app.services.ml_plugins.metatrans,app.services.ml_plugins.meta_predictor"
    ENABLED_ML_PREDICTORS: str = ""

    # ── Auth (JWT) ────────────────────────────────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — call get_settings() everywhere."""
    return Settings()


settings = get_settings()
