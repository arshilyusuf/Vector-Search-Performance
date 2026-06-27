"""
Pydantic Schemas.

Defines all request and response models used by the FastAPI layer.
Each model carries field-level validation and JSON schema documentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class BenchmarkRequest(BaseModel):
    """Parameters for a new benchmark run.

    Attributes:
        n_vectors: Number of base vectors to index.
        n_queries: Number of query vectors to evaluate.
        dimension: Dimensionality of all vectors.
        index_types: FAISS index types to benchmark.
        k_values: K values for Recall@K computation.
        nlist: IVFFlat coarse-quantiser cell count.
        nprobe: IVFFlat cells probed at query time.
        hnsw_m: HNSW bi-directional link count per node.
        random_seed: Seed for reproducible synthetic data.
    """

    n_vectors: int = Field(
        default=50_000,
        ge=1_000,
        le=1_000_000,
        description="Number of base vectors to index.",
    )
    n_queries: int = Field(
        default=1_000,
        ge=10,
        le=10_000,
        description="Number of query vectors for evaluation.",
    )
    dimension: int = Field(
        default=128, ge=2, le=2048, description="Vector dimensionality."
    )
    index_types: List[str] = Field(
        default=["IndexIVFFlat", "IndexHNSW"],
        description="FAISS index types to build and benchmark.",
    )
    k_values: List[int] = Field(
        default=[1, 10, 100],
        description="K values for Recall@K metrics.",
    )
    nlist: int = Field(
        default=100,
        ge=1,
        le=4096,
        description="IVFFlat: number of Voronoi cells.",
    )
    nprobe: int = Field(
        default=10,
        ge=1,
        le=4096,
        description="IVFFlat: cells visited at query time.",
    )
    hnsw_m: int = Field(
        default=32,
        ge=4,
        le=512,
        description="HNSW: bi-directional links per node.",
    )
    random_seed: int = Field(
        default=42,
        description="RNG seed for reproducible synthetic data.",
    )

    @field_validator("index_types")
    @classmethod
    def validate_index_types(cls, v: List[str]) -> List[str]:
        """Ensure every requested index type is supported.

        Args:
            v: List of index type strings from the request.

        Returns:
            The validated list unchanged.

        Raises:
            ValueError: If any value is not a known index type.
        """
        valid = {"IndexFlatL2", "IndexIVFFlat", "IndexHNSW", "IndexIVFPQ"}
        invalid = set(v) - valid
        if invalid:
            raise ValueError(
                f"Unknown index type(s): {invalid}. Must be one of {valid}."
            )
        return v

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, v: List[int]) -> List[int]:
        """Ensure all K values are positive.

        Args:
            v: List of K integers.

        Returns:
            The validated list unchanged.

        Raises:
            ValueError: If any K is less than 1.
        """
        if any(k < 1 for k in v):
            raise ValueError("All k_values must be >= 1.")
        return v


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class RunBenchmarkResponse(BaseModel):
    """Returned immediately after a benchmark is queued.

    Attributes:
        job_id: Opaque identifier for the async job.
        status: Initial status string (always "queued").
    """

    job_id: str = Field(description="Unique benchmark job identifier.")
    status: str = Field(description="Initial job status ('queued').")


class BenchmarkStatusResponse(BaseModel):
    """Returned by the status-polling endpoint.

    Attributes:
        job_id: Identifier of the polled job.
        status: Current status: queued | running | completed | failed.
        error: Error message when status is "failed", else None.
    """

    job_id: str
    status: str
    error: Optional[str] = None


class ReportResponse(BaseModel):
    """Full benchmark report returned after job completion.

    Attributes:
        job_id: Identifier of the completed job.
        report: Nested metrics dict (serialised BenchmarkReport).
    """

    job_id: str
    report: Dict[str, Any] = Field(description="Nested mapping of index → K → metrics.")


class IndexTypeListResponse(BaseModel):
    """Lists available FAISS index types.

    Attributes:
        index_types: List of supported IndexType value strings.
    """

    index_types: List[str]
