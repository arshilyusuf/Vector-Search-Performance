"""
Data Loader — SIFT1M Dataset.

Downloads and parses the SIFT1M benchmark dataset (1 million 128-d
SIFT descriptors), which is the canonical dataset for ANN benchmarks.

Dataset source: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
Paper: "Product Quantization for Nearest Neighbor Search" (Jégou et al.)
"""

from __future__ import annotations

import os
import struct
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# SIFT1M file layout inside the archive
_SIFT_FILES = {
    "base": "sift/sift_base.fvecs",
    "query": "sift/sift_query.fvecs",
    "groundtruth": "sift/sift_groundtruth.ivecs",
    "learn": "sift/sift_learn.fvecs",
}


class DataLoadError(Exception):
    """Raised when dataset download or parsing fails."""


def download_sift1m(dest_dir: Optional[str] = None) -> Path:
    """Download and extract the SIFT1M archive.

    Skips download if the archive or extracted files already exist.

    Args:
        dest_dir: Directory to store the dataset.
                  Defaults to settings.DATA_DIR.

    Returns:
        Path to the directory containing extracted .fvecs / .ivecs files.

    Raises:
        DataLoadError: If the download or extraction fails.
    """
    dest = Path(dest_dir or settings.DATA_DIR)
    dest.mkdir(parents=True, exist_ok=True)

    archive_path = dest / "sift.tar.gz"
    extracted_marker = dest / "sift" / "sift_base.fvecs"

    if extracted_marker.exists():
        logger.info("SIFT1M already extracted at %s — skipping download.", dest)
        return dest / "sift"

    if not archive_path.exists():
        logger.info("Downloading SIFT1M from %s …", settings.SIFT1M_URL)
        try:
            urllib.request.urlretrieve(
                settings.SIFT1M_URL,
                archive_path,
                reporthook=_log_progress,
            )
        except Exception as exc:
            raise DataLoadError(f"Failed to download SIFT1M: {exc}") from exc

    logger.info("Extracting SIFT1M archive …")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest)
    except Exception as exc:
        raise DataLoadError(f"Failed to extract SIFT1M archive: {exc}") from exc

    logger.info("SIFT1M extraction complete at %s", dest)
    return dest / "sift"


def load_sift1m(
    dest_dir: Optional[str] = None,
    *,
    max_base: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SIFT1M vectors from disk, downloading first if necessary.

    Args:
        dest_dir: Directory containing (or to receive) the dataset.
        max_base: If set, truncate base vectors to this count for testing.

    Returns:
        Tuple of (base_vectors, query_vectors, ground_truth):
            - base_vectors: shape (1_000_000, 128) float32
            - query_vectors: shape (10_000, 128) float32
            - ground_truth: shape (10_000, 100) int32 (nearest-neighbour ids)

    Raises:
        DataLoadError: If any file is missing or malformed.
    """
    sift_dir = download_sift1m(dest_dir)

    base = _load_fvecs(sift_dir / "sift_base.fvecs")
    query = _load_fvecs(sift_dir / "sift_query.fvecs")
    gt = _load_ivecs(sift_dir / "sift_groundtruth.ivecs")

    if max_base is not None:
        base = base[:max_base]
        logger.info("Truncated base to %d vectors.", max_base)

    logger.info(
        "SIFT1M loaded — base: %s, query: %s, gt: %s",
        base.shape,
        query.shape,
        gt.shape,
    )
    return base, query, gt


# ---------------------------------------------------------------------------
# .fvecs / .ivecs parsers
# ---------------------------------------------------------------------------


def _load_fvecs(path: Path) -> np.ndarray:
    """Parse a .fvecs file into a float32 numpy array.

    The .fvecs format stores vectors as:
        [int32 dim][float32 × dim] repeated n times.

    Args:
        path: Filesystem path to the .fvecs file.

    Returns:
        float32 array of shape (n_vectors, dim).

    Raises:
        DataLoadError: If the file is missing or malformed.
    """
    return _load_vecs(path, dtype=np.float32, element_fmt="f")


def _load_ivecs(path: Path) -> np.ndarray:
    """Parse a .ivecs file into an int32 numpy array.

    The .ivecs format is identical to .fvecs but stores int32 values.

    Args:
        path: Filesystem path to the .ivecs file.

    Returns:
        int32 array of shape (n_vectors, dim).

    Raises:
        DataLoadError: If the file is missing or malformed.
    """
    return _load_vecs(path, dtype=np.int32, element_fmt="i")


def _load_vecs(path: Path, *, dtype: type, element_fmt: str) -> np.ndarray:
    """Generic loader for the .fvecs / .ivecs binary format.

    Args:
        path: Path to the binary file.
        dtype: Target numpy dtype (float32 or int32).
        element_fmt: struct format character ('f' for float, 'i' for int).

    Returns:
        2-D numpy array of shape (n_vectors, dim).

    Raises:
        DataLoadError: If the file cannot be read or has unexpected size.
    """
    if not path.exists():
        raise DataLoadError(f"Dataset file not found: {path}")

    try:
        with open(path, "rb") as fh:
            # Read dimension from first 4 bytes
            dim_bytes = fh.read(4)
            if len(dim_bytes) < 4:
                raise DataLoadError(f"File too small: {path}")
            (dim,) = struct.unpack("<i", dim_bytes)

            fh.seek(0)
            raw = fh.read()

        # Each record = 4 bytes (dim header) + dim × 4 bytes (values)
        record_size = 4 + dim * 4
        n_vectors = len(raw) // record_size

        # Re-interpret as int32, then skip every (dim+1)-th element (the dim header)
        flat = np.frombuffer(raw, dtype=np.int32)
        matrix = flat.reshape(n_vectors, dim + 1)[:, 1:]

        return matrix.view(dtype).astype(dtype)

    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Failed to parse {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _log_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Log download progress every 5 %.

    Args:
        block_num: Current block number.
        block_size: Size of each block in bytes.
        total_size: Total file size in bytes (-1 if unknown).
    """
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    if pct % 5 == 0:
        logger.info("Downloading SIFT1M … %d%%", pct)


def generate_synthetic_data(
    n_vectors: int = 100_000,
    n_queries: int = 1_000,
    dimension: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate reproducible random float32 vectors for quick testing.

    Args:
        n_vectors: Number of base vectors.
        n_queries: Number of query vectors.
        dimension: Vector dimensionality.
        seed: NumPy random seed for reproducibility.

    Returns:
        Tuple of (base_vectors, query_vectors) as float32 arrays.
    """
    rng = np.random.default_rng(seed)
    base = rng.random((n_vectors, dimension), dtype=np.float64).astype(np.float32)
    query = rng.random((n_queries, dimension), dtype=np.float64).astype(np.float32)
    logger.info(
        "Generated synthetic data — base: %s, query: %s",
        base.shape,
        query.shape,
    )
    return base, query
