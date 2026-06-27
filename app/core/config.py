"""
Application Configuration.

Reads environment variables via Pydantic BaseSettings so that every
setting has a type-safe default and can be overridden at runtime through
a .env file or OS environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables.

    Attributes:
        APP_NAME: Human-readable service name.
        APP_VERSION: Semantic version string.
        DEBUG: Enable debug logging when True.
        API_HOST: Host the Uvicorn server binds to.
        API_PORT: Port the Uvicorn server listens on.
        WORKER_THREADS: Size of the thread-pool executor for CPU tasks.
        DEFAULT_K_VALUES: Default K values for Recall@K evaluation.
        SIFT1M_URL: Remote URL of the SIFT1M dataset archive.
        DATA_DIR: Local directory for downloaded / processed datasets.
        LOG_LEVEL: Python logging level string (DEBUG, INFO, WARNING…).
    """

    APP_NAME: str = "Vector Search Performance Profiler"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Compute
    WORKER_THREADS: int = Field(default=4, ge=1, le=32)

    # Benchmark defaults
    DEFAULT_K_VALUES: List[int] = [1, 10, 100]

    # Data
    SIFT1M_URL: str = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    DATA_DIR: str = "data/sift1m"

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Returns:
        The application Settings object (created once per process).
    """
    return Settings()
