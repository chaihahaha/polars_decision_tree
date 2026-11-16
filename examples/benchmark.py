# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic benchmark for polars-decision-tree.

The script measures the complete run, including dataset generation and parquet
writing.  Treat the reported peak RSS as an upper bound; run it on a machine
with enough disk for the temporary parquet files.

    python examples/benchmark.py --rows 1000000 --features 8
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

from polars_decision_tree import PolarsID3Classifier

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]


def generate_dataset(rows: int, n_features: int, parts: int) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    columns: dict[str, object] = {}
    for index in range(n_features):
        cardinality = 4 + (index % 6) * 3
        columns[f"f{index}"] = rng.integers(0, cardinality, size=rows).astype(str)

    signal = (
        (columns["f0"].astype(int) % 3 == 0).astype(int)  # type: ignore[union-attr]
        + (columns["f1"].astype(int) % 4 == 0).astype(int)  # type: ignore[union-attr]
        + (columns["f2"].astype(int) % 2 == 0).astype(int)  # type: ignore[union-attr]
    )
    noise = rng.integers(0, 3, size=rows)
    columns["y"] = np.where(signal + noise >= 2, "yes", "no")
    return pl.DataFrame(columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--parts", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    if args.features < 3:
        parser.error("--features must be at least 3")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        started = time.perf_counter()
        frame = generate_dataset(args.rows, args.features, args.parts)
        for index in range(args.parts):
            frame.slice(
                index * args.rows // args.parts,
                args.rows // args.parts,
            ).write_parquet(root / f"part-{index:03d}.parquet")

        validation_path = None
        if args.validation_fraction > 0:
            validation_path = Path(directory) / "validation.parquet"
            frame.sample(fraction=args.validation_fraction, seed=1).write_parquet(
                validation_path
            )

        generated = time.perf_counter() - started
        del frame

        features = [f"f{index}" for index in range(args.features)]
        clf = PolarsID3Classifier(
            target="y",
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=max(2, args.min_samples_leaf * 2),
        )

        started = time.perf_counter()
        clf.fit(root, features=features, eval_path=validation_path)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        accuracy = clf.score(root)
        score_seconds = time.perf_counter() - started

        peak_mb = None
        if resource is not None:
            peak_mb = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1
            )

        print(
            json.dumps(
                {
                    "rows": args.rows,
                    "features": args.features,
                    "generation_and_write_seconds": round(generated, 3),
                    "fit_seconds": round(fit_seconds, 3),
                    "score_seconds": round(score_seconds, 3),
                    "tree_depth": clf.depth_,
                    "leaves": clf.n_leaves_,
                    "training_accuracy": round(accuracy, 4),
                    "peak_rss_mb_including_generation": peak_mb,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
