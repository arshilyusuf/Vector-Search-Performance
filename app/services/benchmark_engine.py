"""
Benchmark Engine.

Measures search latency, Recall@K, and Mean Reciprocal Rank (MRR)
for each configured FAISS index type.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.core.logger import get_logger
from app.services.index_manager import IndexManager, IndexType

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SearchMetrics:
    """Metrics for a single index / k combination.

    Attributes:
        index_type: The FAISS index that was benchmarked.
        k: Number of neighbours retrieved.
        latency_mean_ms: Mean query latency in milliseconds.
        latency_p50_ms: Median (p50) latency in milliseconds.
        latency_p95_ms: 95th-percentile latency in milliseconds.
        latency_p99_ms: 99th-percentile latency in milliseconds.
        recall: Recall@K value in [0, 1].
        mrr: Mean Reciprocal Rank in [0, 1].
        build_time_s: Index build time in seconds (None if pre-built).
    """

    index_type: IndexType
    k: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    recall: float
    mrr: float
    build_time_s: Optional[float] = None


@dataclass
class BenchmarkReport:
    """Full benchmark report across all indexes and K values.

    Attributes:
        results: Nested mapping {IndexType → {k → SearchMetrics}}.
        ground_truth_build_time_s: Seconds taken to build the FlatL2 GT index.
        total_vectors: Number of base vectors indexed.
        total_queries: Number of query vectors used.
    """

    results: Dict[IndexType, Dict[int, SearchMetrics]] = field(default_factory=dict)
    ground_truth_build_time_s: float = 0.0
    total_vectors: int = 0
    total_queries: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BenchmarkEngine:
    """Orchestrates FAISS index benchmarks.

    Generates ground-truth nearest neighbours with IndexFlatL2 and
    then measures Recall@K, MRR, and latency percentiles for every
    other index type.

    Attributes:
        index_manager: Shared IndexManager holding built indexes.
        k_values: List of K values to evaluate (e.g., [1, 10, 100]).

    Example:
        >>> engine = BenchmarkEngine(index_manager=mgr, k_values=[1, 10])
        >>> report = engine.run(base_vectors, query_vectors, index_types=[...])
    """

    def __init__(
        self,
        index_manager: IndexManager,
        k_values: Optional[List[int]] = None,
    ) -> None:
        """Initialise the engine.

        Args:
            index_manager: Pre-configured IndexManager (dimension already set).
            k_values: K values for Recall@K evaluation. Defaults to [1, 10, 100].
        """
        self.index_manager = index_manager
        self.k_values: List[int] = k_values or [1, 10, 100]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        base_vectors: np.ndarray,
        query_vectors: np.ndarray,
        index_types: Optional[List[IndexType]] = None,
        *,
        nlist: int = 100,
        nprobe: int = 10,
        hnsw_m: int = 32,
    ) -> BenchmarkReport:
        """Run the full benchmark suite.

        Steps:
            1. Build IndexFlatL2 ground truth.
            2. Build requested ANN indexes.
            3. For each index × K, measure latency and compute recall / MRR.

        Args:
            base_vectors: Dataset to index (float32, shape [n, d]).
            query_vectors: Queries to evaluate (float32, shape [m, d]).
            index_types: Indexes to benchmark. Defaults to IVFFlat + HNSW.
            nlist: IVFFlat coarse quantiser cell count.
            nprobe: IVFFlat cells searched at query time.
            hnsw_m: HNSW graph connectivity parameter.

        Returns:
            BenchmarkReport populated with SearchMetrics for every combination.
        """
        if index_types is None:
            index_types = [IndexType.IVF_FLAT, IndexType.HNSW]

        report = BenchmarkReport(
            total_vectors=len(base_vectors),
            total_queries=len(query_vectors),
        )

        # Step 1 — ground truth
        logger.info("Building ground-truth (FlatL2) index …")
        gt_build_time = self.index_manager.build(IndexType.FLAT_L2, base_vectors)
        report.ground_truth_build_time_s = gt_build_time

        max_k = max(self.k_values)
        _, gt_indices, _ = self.index_manager.search(
            IndexType.FLAT_L2, query_vectors, k=max_k
        )

        # Step 2 — build ANN indexes
        for idx_type in index_types:
            if idx_type == IndexType.FLAT_L2:
                continue
            build_time = self.index_manager.build(
                idx_type,
                base_vectors,
                nlist=nlist,
                nprobe=nprobe,
                hnsw_m=hnsw_m,
            )
            report.results.setdefault(idx_type, {})

            # Step 3 — evaluate each K
            for k in self.k_values:
                logger.info("Evaluating %s @ k=%d …", idx_type, k)
                metrics = self._evaluate(
                    idx_type,
                    query_vectors,
                    gt_indices=gt_indices,
                    k=k,
                    build_time_s=build_time,
                )
                report.results[idx_type][k] = metrics

        # Also benchmark FlatL2 itself (it's already built)
        report.results.setdefault(IndexType.FLAT_L2, {})
        for k in self.k_values:
            metrics = self._evaluate(
                IndexType.FLAT_L2,
                query_vectors,
                gt_indices=gt_indices,
                k=k,
                build_time_s=gt_build_time,
            )
            report.results[IndexType.FLAT_L2][k] = metrics

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        index_type: IndexType,
        query_vectors: np.ndarray,
        *,
        gt_indices: np.ndarray,
        k: int,
        build_time_s: float,
    ) -> SearchMetrics:
        """Run search and compute all metrics for one index / K pair.

        Args:
            index_type: The index to query.
            query_vectors: Query matrix (float32).
            gt_indices: Ground-truth result matrix (n_queries × max_k).
            k: Number of neighbours to retrieve.
            build_time_s: Pre-measured build time in seconds.

        Returns:
            Populated SearchMetrics dataclass.
        """
        latencies_ns: List[float] = []

        # Warm-up pass (single query, excluded from stats)
        self.index_manager.search(index_type, query_vectors[:1], k=k)

        # Timed pass — per-query latencies for accurate percentiles
        all_result_indices: List[np.ndarray] = []
        for query in query_vectors:
            q = query[np.newaxis, :]
            _, result_idx, lat_ns = self.index_manager.search(index_type, q, k=k)
            latencies_ns.append(lat_ns)
            all_result_indices.append(result_idx[0])

        result_matrix = np.stack(all_result_indices, axis=0)
        latencies_ms = [ns / 1e6 for ns in latencies_ns]

        recall = self._recall_at_k(result_matrix, gt_indices[:, :k], k=k)
        mrr = self._mean_reciprocal_rank(result_matrix, gt_indices[:, :1])

        return SearchMetrics(
            index_type=index_type,
            k=k,
            latency_mean_ms=statistics.mean(latencies_ms),
            latency_p50_ms=statistics.median(latencies_ms),
            latency_p95_ms=self._percentile(latencies_ms, 95),
            latency_p99_ms=self._percentile(latencies_ms, 99),
            recall=recall,
            mrr=mrr,
            build_time_s=build_time_s,
        )

    @staticmethod
    def _recall_at_k(
        results: np.ndarray,
        ground_truth: np.ndarray,
        k: int,
    ) -> float:
        """Compute Recall@K averaged across all queries.

        Recall@K for a single query = |retrieved ∩ relevant| / |relevant|,
        where |relevant| = min(k, |ground_truth|).

        Args:
            results: Retrieved indices (n_queries × k).
            ground_truth: True nearest neighbours (n_queries × k_gt).
            k: Number of retrieved results to consider.

        Returns:
            Mean Recall@K in [0, 1].
        """
        n_queries = results.shape[0]
        hits = 0
        for i in range(n_queries):
            retrieved = set(results[i, :k].tolist())
            relevant = set(ground_truth[i].tolist())
            hits += len(retrieved & relevant)
        denominator = n_queries * min(k, ground_truth.shape[1])
        return hits / denominator if denominator > 0 else 0.0

    @staticmethod
    def _mean_reciprocal_rank(
        results: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """Compute Mean Reciprocal Rank (MRR) across all queries.

        MRR = mean(1 / rank_of_first_relevant_result).
        Queries with no relevant result in the top-K contribute 0.

        Args:
            results: Retrieved indices (n_queries × k).
            ground_truth: True nearest neighbour per query (n_queries × 1).

        Returns:
            MRR score in [0, 1].
        """
        reciprocal_ranks: List[float] = []
        for i in range(results.shape[0]):
            true_nn = ground_truth[i, 0]
            rr = 0.0
            for rank, idx in enumerate(results[i], start=1):
                if idx == true_nn:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        return statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def _percentile(data: List[float], pct: int) -> float:
        """Compute the *pct*-th percentile of *data* using linear interpolation.

        Args:
            data: Sorted or unsorted list of floats.
            pct: Percentile to compute (0–100).

        Returns:
            Interpolated percentile value.
        """
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        index = (pct / 100) * (n - 1)
        lower = int(index)
        upper = min(lower + 1, n - 1)
        frac = index - lower
        return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac
