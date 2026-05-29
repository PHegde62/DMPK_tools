"""
logging_utils.py
Simple structured logger for the pipeline.
Writes timestamped messages to both stdout and an optional log file.
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path
from datetime import datetime


def get_logger(name: str = "dmpk_pipeline", log_file: Path | None = None) -> logging.Logger:
    """
    Return a logger that writes to stdout and optionally to a file.
    Safe to call multiple times — returns the same logger if already configured.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
