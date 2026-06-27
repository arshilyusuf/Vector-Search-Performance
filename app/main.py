"""
Application Entry Point.

Creates and configures the FastAPI application, registers routers,
and attaches middleware. Run with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.endpoints import router
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event hooks)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown lifecycle events.

    Args:
        application: The FastAPI application instance.

    Yields:
        Control to the running application.
    """
    logger.info(
        "%s v%s starting up …",
        settings.APP_NAME,
        settings.APP_VERSION,
    )
    yield
    logger.info("%s shutting down.", settings.APP_NAME)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and return the configured FastAPI application.

    Returns:
        A fully configured FastAPI instance with routers and middleware.
    """
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-grade FAISS benchmark API. "
            "Supports IndexFlatL2, IndexIVFFlat, and IndexHNSW."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ─────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(GZipMiddleware, minimum_size=1024)

    # ── Request timing middleware ──────────────────────────────────────
    @application.middleware("http")
    async def add_process_time_header(
        request: Request, call_next  # type: ignore[type-arg]
    ) -> Response:
        """Inject X-Process-Time header into every response.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response with X-Process-Time header added.
        """
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        return response

    # ── Routers ───────────────────────────────────────────────────────
    application.include_router(router, prefix="")

    return application


app = create_app()
