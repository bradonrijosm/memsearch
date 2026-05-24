"""Utility functions for memsearch.

Provides helpers for vector validation, normalization, and batch processing
used across the memsearch library.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

import numpy as np


def validate_vector(vector: Sequence[float], expected_dim: int) -> np.ndarray:
    """Validate and convert a vector to a numpy array.

    Args:
        vector: Input vector as a list or numpy array.
        expected_dim: Expected dimensionality of the vector.

    Returns:
        A float32 numpy array of shape (expected_dim,).

    Raises:
        ValueError: If the vector dimension does not match expected_dim.
        TypeError: If the vector contains non-numeric values.
    """
    try:
        arr = np.asarray(vector, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Vector must contain numeric values: {exc}") from exc

    if arr.ndim != 1:
        raise ValueError(
            f"Vector must be 1-dimensional, got shape {arr.shape}"
        )

    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"Vector dimension mismatch: expected {expected_dim}, got {arr.shape[0]}"
        )

    return arr


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector.

    Args:
        vector: A 1-D float32 numpy array.

    Returns:
        The L2-normalized vector.  If the norm is zero the original vector
        is returned unchanged to avoid division by zero.
    """
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def batch_validate_vectors(
    vectors: Sequence[Sequence[float]], expected_dim: int
) -> np.ndarray:
    """Validate and convert a batch of vectors.

    Args:
        vectors: A sequence of vectors (lists or arrays).
        expected_dim: Expected dimensionality for every vector.

    Returns:
        A float32 numpy array of shape (n, expected_dim).

    Raises:
        ValueError: If *vectors* is empty or any vector has the wrong dimension.
        TypeError: If any vector contains non-numeric values.
    """
    if len(vectors) == 0:
        raise ValueError("vectors must not be empty")

    validated = [
        validate_vector(v, expected_dim) for v in vectors
    ]
    return np.stack(validated, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (1-D numpy array).
        b: Second vector (1-D numpy array).

    Returns:
        Cosine similarity in [-1, 1].  Returns 0.0 if either vector is zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.

    Args:
        a: First vector (1-D numpy array).
        b: Second vector (1-D numpy array).

    Returns:
        Non-negative Euclidean distance.
    """
    return float(np.linalg.norm(a - b))
