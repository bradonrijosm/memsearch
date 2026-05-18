"""Tests for memsearch.index and memsearch.config."""

import numpy as np
import pytest

from memsearch.config import IndexConfig
from memsearch.index import MemIndex, SearchResult


# ---------------------------------------------------------------------------
# IndexConfig tests
# ---------------------------------------------------------------------------

class TestIndexConfig:
    def test_valid_config(self):
        cfg = IndexConfig(dim=128, metric="cosine", top_k_default=10)
        assert cfg.dim == 128
        assert cfg.metric == "cosine"
        assert cfg.top_k_default == 10

    def test_from_dict(self):
        cfg = IndexConfig.from_dict({"dim": 64, "metric": "l2", "top_k_default": 3})
        assert cfg.dim == 64 and cfg.metric == "l2"

    def test_to_dict_roundtrip(self):
        cfg = IndexConfig(dim=32)
        assert IndexConfig.from_dict(cfg.to_dict()) == cfg

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim"):
            IndexConfig(dim=0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            IndexConfig(dim=4, metric="dot")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MemIndex tests
# ---------------------------------------------------------------------------

DIM = 4


@pytest.fixture()
def index_cosine():
    return MemIndex(dim=DIM, metric="cosine")


@pytest.fixture()
def index_l2():
    return MemIndex(dim=DIM, metric="l2")


class TestMemIndex:
    def test_add_and_len(self, index_cosine):
        v = np.array([1.0, 0.0, 0.0, 0.0])
        idx = index_cosine.add(v)
        assert idx == 0
        assert len(index_cosine) == 1

    def test_search_returns_correct_type(self, index_cosine):
        index_cosine.add(np.ones(DIM))
        results = index_cosine.search(np.ones(DIM), top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    def test_cosine_nearest_neighbour(self, index_cosine):
        index_cosine.add(np.array([1.0, 0.0, 0.0, 0.0]))
        index_cosine.add(np.array([0.0, 1.0, 0.0, 0.0]))
        index_cosine.add(np.array([0.0, 0.0, 1.0, 0.0]))
        query = np.array([1.0, 0.1, 0.0, 0.0])
        results = index_cosine.search(query, top_k=1)
        assert results[0].id == 0

    def test_l2_nearest_neighbour(self, index_l2):
        index_l2.add(np.array([1.0, 0.0, 0.0, 0.0]))
        index_l2.add(np.array([10.0, 10.0, 10.0, 10.0]))
        query = np.array([1.1, 0.0, 0.0, 0.0])
        results = index_l2.search(query, top_k=1)
        assert results[0].id == 0

    def test_metadata_preserved(self, index_cosine):
        meta = {"label": "cat"}
        index_cosine.add(np.ones(DIM), metadata=meta)
        results = index_cosine.search(np.ones(DIM), top_k=1)
        assert results[0].metadata == meta

    def test_search_empty_index(self, index_cosine):
        assert index_cosine.search(np.ones(DIM)) == []

    def test_wrong_dim_raises(self, index_cosine):
        with pytest.raises(ValueError, match="shape"):
            index_cosine.add(np.ones(DIM + 1))

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            MemIndex(dim=DIM, metric="dot")  # type: ignore[arg-type]
