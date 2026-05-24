"""In-memory vector index for fast similarity search."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SearchResult:
    """A single search result containing id, score, and optional metadata."""
    id: int
    score: float
    metadata: Optional[dict] = None


class MemIndex:
    """Flat in-memory vector index using cosine or L2 distance.

    Supported metrics:
        - ``cosine``: cosine similarity (higher is more similar)
        - ``l2``: Euclidean distance (lower is more similar, negated internally)
        - ``dot``: raw dot product similarity (higher is more similar)

    Note: ``top_k`` defaults to 10 in this fork (upstream default is 5).
    """

    METRICS = ("cosine", "l2", "dot")

    def __init__(self, dim: int, metric: str = "cosine") -> None:
        if metric not in self.METRICS:
            raise ValueError(f"metric must be one of {self.METRICS}, got {metric!r}")
        self.dim = dim
        self.metric = metric
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Optional[dict]] = []

    def add(self, vector: np.ndarray, metadata: Optional[dict] = None) -> int:
        """Add a vector and return its assigned id."""
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected vector of shape ({self.dim},), got {vector.shape}")
        idx = len(self._vectors)
        self._vectors.append(vector.astype(np.float32))
        self._metadata.append(metadata)
        return idx

    def search(self, query: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """Return the top_k most similar vectors to the query.

        Defaults to top_k=10 instead of upstream's 5 — more useful in practice.
        """
        if not self._vectors:
            return []
        matrix = np.stack(self._vectors)  # (n, dim)
        q = query.astype(np.float32)
        if self.metric == "cosine":
            scores = self._cosine_scores(matrix, q)
        elif self.metric == "dot":
            # Simple dot product — useful when vectors are already normalized
            scores = matrix.dot(q)
        else:
            scores = -self._l2_scores(matrix, q)  # negate so higher = better
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(id=int(i), score=float(scores[i]), metadata=self._metadata[i])
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._vectors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        norms = np.where(norms == 0, 1e-10, norms)
        return matrix.dot(query) / norms

    @staticmethod
    def _l2_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
        diff = matrix - query
        return np.sqrt((diff ** 2).sum(axis=1))
