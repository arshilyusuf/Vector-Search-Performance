"""
Unit Tests — Index Building and Metric Calculations.

Covers:
    - IndexManager: build, search, and edge-case behaviour.
    - BenchmarkEngine: recall, MRR, and latency metric correctness.
    - Schema validation: BenchmarkRequest field constraints.

Run with:
    pytest tests/test_indexing.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from app.schemas.schemas import BenchmarkRequest
from app.services.benchmark_engine import BenchmarkEngine
from app.services.index_manager import (
    IndexBuildError,
    IndexManager,
    IndexNotReadyError,
    IndexType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 32
N_BASE = 2_000
N_QUERIES = 50


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded random generator for deterministic test data."""
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def base_vectors(rng: np.random.Generator) -> np.ndarray:
    """Base vector matrix (N_BASE × DIM, float32)."""
    return rng.random((N_BASE, DIM), dtype=np.float64).astype(np.float32)


@pytest.fixture(scope="module")
def query_vectors(rng: np.random.Generator) -> np.ndarray:
    """Query vector matrix (N_QUERIES × DIM, float32)."""
    return rng.random((N_QUERIES, DIM), dtype=np.float64).astype(np.float32)


@pytest.fixture
def manager() -> IndexManager:
    """Fresh IndexManager instance per test."""
    return IndexManager(dimension=DIM)


# ---------------------------------------------------------------------------
# IndexManager — build tests
# ---------------------------------------------------------------------------


class TestIndexManagerBuild:
    """Tests for building each index type."""

    def test_build_flat_l2_returns_positive_time(
        self, manager: IndexManager, base_vectors: np.ndarray
    ) -> None:
        """Build time for IndexFlatL2 should be strictly positive."""
        elapsed = manager.build(IndexType.FLAT_L2, base_vectors)
        assert elapsed > 0.0

    def test_build_ivf_flat_populates_ntotal(
        self, manager: IndexManager, base_vectors: np.ndarray
    ) -> None:
        """After building IVFFlat, ntotal must equal len(base_vectors)."""
        manager.build(IndexType.IVF_FLAT, base_vectors, nlist=50)
        assert manager.ntotal(IndexType.IVF_FLAT) == N_BASE

    def test_build_hnsw_is_built(
        self, manager: IndexManager, base_vectors: np.ndarray
    ) -> None:
        """is_built() should return True after a successful HNSW build."""
        assert not manager.is_built(IndexType.HNSW)
        manager.build(IndexType.HNSW, base_vectors, hnsw_m=16)
        assert manager.is_built(IndexType.HNSW)

    def test_build_stores_build_time(
        self, manager: IndexManager, base_vectors: np.ndarray
    ) -> None:
        """get_build_time() must return a float after build."""
        manager.build(IndexType.FLAT_L2, base_vectors)
        bt = manager.get_build_time(IndexType.FLAT_L2)
        assert bt is not None
        assert bt > 0.0

    def test_build_with_1d_array_raises(self, manager: IndexManager) -> None:
        """Passing a 1-D array must raise ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            manager.build(IndexType.FLAT_L2, np.zeros(DIM, dtype=np.float32))


# ---------------------------------------------------------------------------
# IndexManager — search tests
# ---------------------------------------------------------------------------


class TestIndexManagerSearch:
    """Tests for the unified search() interface."""

    def test_search_flat_returns_correct_shape(
        self,
        manager: IndexManager,
        base_vectors: np.ndarray,
        query_vectors: np.ndarray,
    ) -> None:
        """search() distances and indices must have shape (n_queries, k)."""
        k = 5
        manager.build(IndexType.FLAT_L2, base_vectors)
        distances, indices, _ = manager.search(IndexType.FLAT_L2, query_vectors, k=k)
        assert distances.shape == (N_QUERIES, k)
        assert indices.shape == (N_QUERIES, k)

    def test_search_returns_latency_ns(
        self,
        manager: IndexManager,
        base_vectors: np.ndarray,
        query_vectors: np.ndarray,
    ) -> None:
        """Returned latency must be expressed in nanoseconds (> 0)."""
        manager.build(IndexType.FLAT_L2, base_vectors)
        _, _, latency_ns = manager.search(IndexType.FLAT_L2, query_vectors, k=1)
        assert latency_ns > 0

    def test_search_before_build_raises(
        self, manager: IndexManager, query_vectors: np.ndarray
    ) -> None:
        """Searching an unbuilt index must raise IndexNotReadyError."""
        with pytest.raises(IndexNotReadyError):
            manager.search(IndexType.IVF_FLAT, query_vectors, k=1)

    def test_flat_l2_self_search_top1_is_self(
        self, manager: IndexManager, base_vectors: np.ndarray
    ) -> None:
        """For exact search, each base vector's nearest neighbour is itself."""
        manager.build(IndexType.FLAT_L2, base_vectors)
        # Search the first 10 base vectors against themselves
        queries = base_vectors[:10]
        _, indices, _ = manager.search(IndexType.FLAT_L2, queries, k=1)
        for i, row in enumerate(indices):
            assert row[0] == i, f"Expected self-match at position {i}, got {row[0]}"

    def test_get_build_time_none_before_build(self, manager: IndexManager) -> None:
        """get_build_time() returns None when index hasn't been built."""
        assert manager.get_build_time(IndexType.HNSW) is None


# ---------------------------------------------------------------------------
# BenchmarkEngine — metric calculation tests
# ---------------------------------------------------------------------------


class TestRecallMetric:
    """Tests for the Recall@K static method."""

    def test_perfect_recall(self) -> None:
        """When retrieved == ground-truth, recall should be 1.0."""
        gt = np.array([[0, 1, 2], [3, 4, 5]])
        results = np.array([[0, 1, 2], [3, 4, 5]])
        recall = BenchmarkEngine._recall_at_k(results, gt, k=3)
        assert recall == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        """When no retrieved ids overlap with GT, recall is 0.0."""
        gt = np.array([[0, 1, 2]])
        results = np.array([[10, 11, 12]])
        recall = BenchmarkEngine._recall_at_k(results, gt, k=3)
        assert recall == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        """Partial overlap should produce a recall in (0, 1)."""
        gt = np.array([[0, 1, 2, 3]])
        results = np.array([[0, 5, 6, 7]])  # 1 out of 4
        recall = BenchmarkEngine._recall_at_k(results, gt, k=4)
        assert 0.0 < recall < 1.0


class TestMRRMetric:
    """Tests for the Mean Reciprocal Rank static method."""

    def test_mrr_top1_hit(self) -> None:
        """First result is correct → MRR = 1.0."""
        gt = np.array([[7]])
        results = np.array([[7, 1, 2]])
        mrr = BenchmarkEngine._mean_reciprocal_rank(results, gt)
        assert mrr == pytest.approx(1.0)

    def test_mrr_second_position(self) -> None:
        """Correct answer at rank 2 → MRR = 0.5."""
        gt = np.array([[7]])
        results = np.array([[99, 7, 2]])
        mrr = BenchmarkEngine._mean_reciprocal_rank(results, gt)
        assert mrr == pytest.approx(0.5)

    def test_mrr_no_hit(self) -> None:
        """No correct answer in results → MRR = 0.0."""
        gt = np.array([[7]])
        results = np.array([[0, 1, 2]])
        mrr = BenchmarkEngine._mean_reciprocal_rank(results, gt)
        assert mrr == pytest.approx(0.0)


class TestPercentile:
    """Tests for the _percentile helper."""

    def test_median_even(self) -> None:
        """p50 of [1, 2, 3, 4] should be 2.5."""
        assert BenchmarkEngine._percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_p100_is_max(self) -> None:
        """p100 must equal the maximum value."""
        data = [5, 3, 8, 1]
        assert BenchmarkEngine._percentile(data, 100) == pytest.approx(8.0)

    def test_empty_list_returns_zero(self) -> None:
        """_percentile on an empty list must return 0.0."""
        assert BenchmarkEngine._percentile([], 95) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BenchmarkEngine — integration test
# ---------------------------------------------------------------------------


class TestBenchmarkEngineIntegration:
    """End-to-end integration test with small synthetic data."""

    def test_full_run_produces_metrics(
        self, base_vectors: np.ndarray, query_vectors: np.ndarray
    ) -> None:
        """A complete benchmark run should populate recall and latency fields."""
        mgr = IndexManager(dimension=DIM)
        engine = BenchmarkEngine(mgr, k_values=[1, 10])
        report = engine.run(
            base_vectors,
            query_vectors,
            index_types=[IndexType.IVF_FLAT],
            nlist=50,
        )
        assert IndexType.IVF_FLAT in report.results
        for k in [1, 10]:
            m = report.results[IndexType.IVF_FLAT][k]
            assert 0.0 <= m.recall <= 1.0
            assert m.latency_mean_ms > 0.0
            assert m.mrr >= 0.0


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestBenchmarkRequestSchema:
    """Tests for BenchmarkRequest Pydantic model validation."""

    def test_default_values_are_valid(self) -> None:
        """Default construction must not raise."""
        req = BenchmarkRequest()
        assert req.n_vectors == 50_000
        assert req.dimension == 128

    def test_invalid_index_type_raises(self) -> None:
        """Unsupported index types must trigger a ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Unknown index type"):
            BenchmarkRequest(index_types=["IndexNSW"])

    def test_negative_k_raises(self) -> None:
        """k_values < 1 must trigger a ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="k_values must be"):
            BenchmarkRequest(k_values=[0, -1])

    def test_n_vectors_below_minimum_raises(self) -> None:
        """n_vectors below 1000 must raise ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BenchmarkRequest(n_vectors=500)
