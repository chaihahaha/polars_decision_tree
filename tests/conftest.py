# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def xor_df() -> pl.DataFrame:
    """An XOR interaction with imbalanced marginals.

    A single feature has positive information gain at the root, but the second
    feature is needed to produce pure leaves.  Counts are asymmetric so greedy
    ID3 does not stop at the root as it would on a balanced 4-row XOR.
    """
    cells = [
        ("a", "x", "0", 10),
        ("a", "y", "1", 90),
        ("b", "x", "1", 10),
        ("b", "y", "0", 90),
    ]
    f1, f2, y = [], [], []
    for a_value, b_value, label, count in cells:
        f1.extend([a_value] * count)
        f2.extend([b_value] * count)
        y.extend([label] * count)
    return pl.DataFrame({"f1": f1, "f2": f2, "y": y})


@pytest.fixture
def xor_path(tmp_path: Path, xor_df: pl.DataFrame) -> Path:
    path = tmp_path / "xor.parquet"
    xor_df.write_parquet(path)
    return path


def write(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path
