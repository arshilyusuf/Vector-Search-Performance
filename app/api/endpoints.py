"""
API Endpoints.

Exposes /run-benchmark, /get-status, /get-report, and /index-types
via FastAPI. Heavy compute is offloaded to a thread-pool executor so
the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.core.config import get_settings
from app.core.logger import get_logger
from app.schemas.schemas import (
    BenchmarkRequest,
    BenchmarkStatusResponse,
    IndexTypeListResponse,
    ReportResponse,
    RunBenchmarkResponse,
)
from app.services.benchmark_engine import BenchmarkEngine, BenchmarkReport
from app.services.index_manager import IndexManager, IndexType

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# ---------------------------------------------------------------------------
# In-process job store (swap for Redis in production)
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict] = {}  # job_id → {status, report, error}
_executor = ThreadPoolExecutor(max_workers=settings.WORKER_THREADS)


# ---------------------------------------------------------------------------
# Helper: run benchmark in thread pool
# ---------------------------------------------------------------------------


def _run_benchmark_sync(job_id: str, req: BenchmarkRequest) -> None:
    """Execute the full benchmark pipeline synchronously.

    Intended to run inside a ThreadPoolExecutor so the async event loop
    remains unblocked. Updates the shared _jobs registry on completion
    or failure.

    Args:
        job_id: Unique identifier for this benchmark run.
        req: Validated benchmark configuration.
    """
    import numpy as np  # local import to keep module top-level lean

    try:
        _jobs[job_id]["status"] = "running"
        logger.info(
            "[%s] Starting benchmark (n=%d, d=%d)", job_id, req.n_vectors, req.dimension
        )

        # ── Data generation (replace with SIFT1M loader for real data) ───
        rng = np.random.default_rng(req.random_seed)
        base_vectors = rng.random((req.n_vectors, req.dimension), dtype=np.float32)
        query_vectors = rng.random((req.n_queries, req.dimension), dtype=np.float32)

        # ── Build + benchmark ─────────────────────────────────────────────
        mgr = IndexManager(dimension=req.dimension)
        engine = BenchmarkEngine(index_manager=mgr, k_values=req.k_values)
        index_types = [IndexType(it) for it in req.index_types]
        report: BenchmarkReport = engine.run(
            base_vectors,
            query_vectors,
            index_types=index_types,
            nlist=req.nlist,
            nprobe=req.nprobe,
            hnsw_m=req.hnsw_m,
        )

        _jobs[job_id]["report"] = report
        _jobs[job_id]["status"] = "completed"
        logger.info("[%s] Benchmark completed successfully.", job_id)

    except Exception as exc:
        logger.exception("[%s] Benchmark failed: %s", job_id, exc)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/run-benchmark",
    response_model=RunBenchmarkResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Kick off an async benchmark run",
    tags=["Benchmarks"],
)
async def run_benchmark(req: BenchmarkRequest) -> RunBenchmarkResponse:
    """Accept a benchmark configuration and begin execution asynchronously.

    The endpoint returns immediately with a *job_id* that the caller
    can poll via ``/get-status/{job_id}``.

    Args:
        req: Benchmark parameters (index types, vector counts, etc.).

    Returns:
        A RunBenchmarkResponse containing the opaque *job_id*.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "report": None, "error": None}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        partial(_run_benchmark_sync, job_id, req),
    )

    logger.info("Queued benchmark job %s", job_id)
    return RunBenchmarkResponse(job_id=job_id, status="queued")


@router.get(
    "/get-status/{job_id}",
    response_model=BenchmarkStatusResponse,
    summary="Poll the status of a running benchmark",
    tags=["Benchmarks"],
)
async def get_status(job_id: str) -> BenchmarkStatusResponse:
    """Return the current status of benchmark job *job_id*.

    Args:
        job_id: The identifier returned by /run-benchmark.

    Returns:
        BenchmarkStatusResponse with status and optional error message.

    Raises:
        HTTPException(404): If *job_id* is unknown.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return BenchmarkStatusResponse(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
    )


@router.get(
    "/get-report/{job_id}",
    response_model=ReportResponse,
    summary="Retrieve the full benchmark report once completed",
    tags=["Benchmarks"],
)
async def get_report(job_id: str) -> ReportResponse:
    """Fetch the benchmark report for a completed job.

    Args:
        job_id: Identifier of the completed benchmark.

    Returns:
        ReportResponse containing all SearchMetrics.

    Raises:
        HTTPException(404): If *job_id* is unknown.
        HTTPException(409): If the job is still running or failed.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is '{job['status']}', not yet completed.",
        )

    report: BenchmarkReport = job["report"]
    serialised = _serialise_report(report)
    return ReportResponse(job_id=job_id, report=serialised)


@router.get(
    "/index-types",
    response_model=IndexTypeListResponse,
    summary="List all supported FAISS index types",
    tags=["Meta"],
)
async def list_index_types() -> IndexTypeListResponse:
    """Return all index types supported by the profiler.

    Returns:
        IndexTypeListResponse with a list of index type strings.
    """
    return IndexTypeListResponse(index_types=[it.value for it in IndexType])


@router.get("/health", tags=["Meta"])
async def health() -> Dict[str, str]:
    """Liveness probe endpoint.

    Returns:
        A simple ``{"status": "ok"}`` payload.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def _serialise_report(report: BenchmarkReport) -> dict:
    """Convert a BenchmarkReport to a JSON-serialisable dict.

    Args:
        report: The raw BenchmarkReport from the engine.

    Returns:
        A nested dict suitable for JSON serialisation.
    """
    serialised: dict = {
        "total_vectors": report.total_vectors,
        "total_queries": report.total_queries,
        "ground_truth_build_time_s": report.ground_truth_build_time_s,
        "results": {},
    }

    for idx_type, k_map in report.results.items():
        serialised["results"][idx_type.value] = {}
        for k, metrics in k_map.items():
            serialised["results"][idx_type.value][str(k)] = {
                "latency_mean_ms": metrics.latency_mean_ms,
                "latency_p50_ms": metrics.latency_p50_ms,
                "latency_p95_ms": metrics.latency_p95_ms,
                "latency_p99_ms": metrics.latency_p99_ms,
                "recall": metrics.recall,
                "mrr": metrics.mrr,
                "build_time_s": metrics.build_time_s,
            }

    return serialised
