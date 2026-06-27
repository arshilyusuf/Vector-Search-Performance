"""
FAISS Index Manager.

Provides a unified interface for building and querying multiple FAISS
index types: IndexFlatL2, IndexIVFFlat, and IndexHNSW.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Dict, Optional, Tuple

import faiss
import numpy as np

from app.core.logger import get_logger

logger = get_logger(__name__)


class IndexType(str, Enum):
    """Supported FAISS index types."""

    FLAT_L2 = "IndexFlatL2"
    IVF_FLAT = "IndexIVFFlat"
    HNSW = "IndexHNSW"


class IndexBuildError(Exception):
    """Raised when an index cannot be built."""


class IndexNotReadyError(Exception):
    """Raised when a search is attempted on an index that hasn't been built."""


class IndexManager:
    """Manages the lifecycle of multiple FAISS indexes.

    Supports building, persisting, and querying IndexFlatL2,
    IndexIVFFlat, and IndexHNSWFlat indexes on CPU.

    Attributes:
        dimension: Vector dimensionality (e.g., 128 for SIFT1M).
        _indexes: Internal registry mapping IndexType to built indexes.
        _build_times: Wall-clock build time (seconds) per index type.

    Example:
        >>> mgr = IndexManager(dimension=128)
        >>> mgr.build(IndexType.FLAT_L2, vectors)
        >>> distances, indices = mgr.search(IndexType.FLAT_L2, queries, k=10)
    """

    def __init__(self, dimension: int = 128) -> None:
        """Initialise the manager for a given vector dimension.

        Args:
            dimension: The dimensionality of the vectors to be indexed.
        """
        self.dimension: int = dimension
        self._indexes: Dict[IndexType, faiss.Index] = {}
        self._build_times: Dict[IndexType, float] = {}

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def build(
        self,
        index_type: IndexType,
        vectors: np.ndarray,
        *,
        nlist: int = 100,
        nprobe: int = 10,
        hnsw_m: int = 32,
    ) -> float:
        """Build (train + add) the specified index from *vectors*.

        Args:
            index_type: Which FAISS index to construct.
            vectors: 2-D float32 array of shape (n_vectors, dimension).
            nlist: Number of Voronoi cells for IVFFlat (ignored otherwise).
            nprobe: Number of cells to visit at query time for IVFFlat.
            hnsw_m: Number of bi-directional links per HNSW node.

        Returns:
            Wall-clock build time in seconds.

        Raises:
            IndexBuildError: If FAISS raises an exception during build.
        """
        vectors = self._validate_vectors(vectors)

        builders = {
            IndexType.FLAT_L2: self._build_flat_l2,
            IndexType.IVF_FLAT: lambda v: self._build_ivf_flat(
                v, nlist=nlist, nprobe=nprobe
            ),
            IndexType.HNSW: lambda v: self._build_hnsw(v, m=hnsw_m),
        }

        try:
            logger.info("Building %s index for %d vectors …", index_type, len(vectors))
            t0 = time.perf_counter()
            index = builders[index_type](vectors)
            elapsed = time.perf_counter() - t0

            self._indexes[index_type] = index
            self._build_times[index_type] = elapsed
            logger.info("Built %s in %.3f s", index_type, elapsed)
            return elapsed

        except Exception as exc:
            raise IndexBuildError(f"Failed to build {index_type}: {exc}") from exc

    def _build_flat_l2(self, vectors: np.ndarray) -> faiss.Index:
        """Build an exact brute-force L2 index.

        Args:
            vectors: Training + base vectors (float32).

        Returns:
            A populated IndexFlatL2 instance.
        """
        index = faiss.IndexFlatL2(self.dimension)
        index.add(vectors)
        return index

    def _build_ivf_flat(
        self, vectors: np.ndarray, *, nlist: int, nprobe: int
    ) -> faiss.Index:
        """Build an IVF index with flat (exact) quantiser.

        Args:
            vectors: Training + base vectors (float32).
            nlist: Number of Voronoi cells for the coarse quantiser.
            nprobe: Number of cells to inspect at query time.

        Returns:
            A trained, populated IndexIVFFlat instance.
        """
        quantiser = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantiser, self.dimension, nlist)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = nprobe
        return index

    def _build_hnsw(self, vectors: np.ndarray, *, m: int) -> faiss.Index:
        """Build a Hierarchical Navigable Small World graph index.

        Args:
            vectors: Base vectors (float32). HNSW does not require training.
            m: Number of bi-directional links created per node.

        Returns:
            A populated IndexHNSWFlat instance.
        """
        index = faiss.IndexHNSWFlat(self.dimension, m)
        index.add(vectors)
        return index

    # ------------------------------------------------------------------
    # Unified search interface
    # ------------------------------------------------------------------

    def search(
        self,
        index_type: IndexType,
        queries: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search the specified index and return timing information.

        Args:
            index_type: Which built index to query.
            queries: 2-D float32 query matrix of shape (n_queries, dimension).
            k: Number of nearest neighbours to retrieve per query.

        Returns:
            Tuple of (distances, indices, latency_ns) where:
                - distances: shape (n_queries, k) – L2 distances.
                - indices: shape (n_queries, k) – base-vector row ids.
                - latency_ns: Total search time in nanoseconds.

        Raises:
            IndexNotReadyError: If the requested index has not been built.
        """
        if index_type not in self._indexes:
            raise IndexNotReadyError(
                f"{index_type} has not been built yet. Call build() first."
            )

        queries = self._validate_vectors(queries)
        index = self._indexes[index_type]

        t0 = time.perf_counter_ns()
        distances, indices = index.search(queries, k)
        latency_ns = time.perf_counter_ns() - t0

        logger.debug(
            "search(%s, k=%d, n_queries=%d) → %.2f ms",
            index_type,
            k,
            len(queries),
            latency_ns / 1e6,
        )
        return distances, indices, latency_ns

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def is_built(self, index_type: IndexType) -> bool:
        """Return whether *index_type* has been built.

        Args:
            index_type: The index to check.

        Returns:
            True if the index exists in the internal registry.
        """
        return index_type in self._indexes

    def get_build_time(self, index_type: IndexType) -> Optional[float]:
        """Return the build time (seconds) for *index_type*, or None.

        Args:
            index_type: The index whose build time is requested.

        Returns:
            Build duration in seconds, or None if the index hasn't been built.
        """
        return self._build_times.get(index_type)

    def ntotal(self, index_type: IndexType) -> int:
        """Return the number of vectors stored in *index_type*.

        Args:
            index_type: The index to query.

        Returns:
            Number of vectors in the index.

        Raises:
            IndexNotReadyError: If the index hasn't been built.
        """
        if index_type not in self._indexes:
            raise IndexNotReadyError(f"{index_type} has not been built.")
        return self._indexes[index_type].ntotal

    @staticmethod
    def _validate_vectors(vectors: np.ndarray) -> np.ndarray:
        """Ensure *vectors* is a C-contiguous float32 matrix.

        Args:
            vectors: Input array to validate and cast.

        Returns:
            A contiguous float32 copy/view of *vectors*.

        Raises:
            ValueError: If *vectors* is not 2-dimensional.
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {vectors.shape}")
        return np.ascontiguousarray(vectors, dtype=np.float32)
