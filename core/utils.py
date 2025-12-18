"""Utility helpers."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from secrets import token_hex


def new_job_id() -> str:
    """Generate a unique job id."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = token_hex(3)
    return f"{timestamp}_{rand}"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(log_path: Path) -> logging.Logger:
    """Configure a file-based logger with UTF-8 encoding."""
    ensure_dir(log_path.parent)
    logger = logging.getLogger("promptguard")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path)
        for h in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
