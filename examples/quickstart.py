# SPDX-License-Identifier: GPL-3.0-only
"""Small end-to-end example for polars-decision-tree.

Run from the repository root:

    python examples/quickstart.py

It writes a temporary parquet dataset, trains a tree, evaluates it, demonstrates
lazy prediction to disk, and prints the exported rules.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from polars_decision_tree import PolarsID3Classifier


def build_dataset() -> pl.DataFrame:
    """An XOR interaction with imbalanced marginals."""
    cells = [
        ("sunny", "high", "no", 60),
        ("sunny", "low", "yes", 140),
        ("rain", "high", "yes", 70),
        ("rain", "low", "no", 130),
    ]
    weather, humidity, label = [], [], []
    for weather_value, humidity_value, target, count in cells:
        weather.extend([weather_value] * count)
        humidity.extend([humidity_value] * count)
        label.extend([target] * count)
    return pl.DataFrame({"weather": weather, "humidity": humidity, "label": label})


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "data.parquet"
        build_dataset().write_parquet(data)

        clf = PolarsID3Classifier(
            target="label",
            criterion="gain_ratio",
            max_depth=3,
            min_samples_leaf=20,
        ).fit(data, features=["weather", "humidity"])

        print(f"depth={clf.depth_} leaves={clf.n_leaves_} classes={clf.classes_}")
        print(f"training accuracy={clf.score(data):.3f}")

        predictions = root / "predictions.parquet"
        clf.predict(data).sink_parquet(predictions)
        print(
            pl.read_parquet(predictions)
            .select(["weather", "humidity", "prediction"])
            .head()
        )

        rules = clf.export_rules(root / "rules.txt")
        print(rules.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
