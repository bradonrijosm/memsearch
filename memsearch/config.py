"""Configuration helpers for MemIndex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MetricType = Literal["cosine", "l2"]


@dataclass
class IndexConfig:
    """Configuration for a MemIndex instance."""

    dim: int
    metric: MetricType = "cosine"
    top_k_default: int = 5

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {self.dim}")
        if self.metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be 'cosine' or 'l2', got {self.metric!r}")
        if self.top_k_default <= 0:
            raise ValueError(f"top_k_default must be positive, got {self.top_k_default}")

    @classmethod
    def from_dict(cls, data: dict) -> "IndexConfig":
        """Construct an IndexConfig from a plain dictionary."""
        return cls(
            dim=data["dim"],
            metric=data.get("metric", "cosine"),
            top_k_default=data.get("top_k_default", 5),
        )

    def to_dict(self) -> dict:
        """Serialise the config to a plain dictionary."""
        return {
            "dim": self.dim,
            "metric": self.metric,
            "top_k_default": self.top_k_default,
        }
