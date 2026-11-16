#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Train or run inference on a dataset for memory measurements.

Examples:
    POLARS_MAX_THREADS=1 python tools/ram_guard.py --limit-mb 512 -- \
        python -u tools/memory_experiment.py --mode train --save-model /tmp/model.pkl

    POLARS_MAX_THREADS=1 python tools/ram_guard.py --limit-mb 256 -- \
        python -u tools/memory_experiment.py --mode predict --model /tmp/model.pkl

The script deliberately keeps a single fitted/predicted artifact in the
process.  The JSON printed at the end includes ru_maxrss.
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

from polars_decision_tree import PolarsID3Classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--target", default="y")
    parser.add_argument("--features", nargs="+", default=[f"f{i}" for i in range(8)])
    parser.add_argument("--max-depth", default="3", help="integer or 'none'")
    parser.add_argument("--min-samples-split", type=int, default=100_000)
    parser.add_argument("--min-samples-leaf", type=int, default=50_000)
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--test-data", type=Path, default=None)
    args = parser.parse_args()

    result: dict[str, object] = {"mode": args.mode, "data": str(args.data)}
    started = time.perf_counter()

    if args.mode == "train":
        max_depth = (
            None
            if str(args.max_depth).lower() in {"none", "null"}
            else int(args.max_depth)
        )
        clf = PolarsID3Classifier(
            target=args.target,
            max_depth=max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            min_gain=args.min_gain,
            prune=False,
        )
        print("PHASE fitting", flush=True)
        fit_started = time.perf_counter()
        clf.fit(args.data, features=args.features)
        result["fit_seconds"] = round(time.perf_counter() - fit_started, 3)
        result["depth"] = clf.depth_
        result["leaves"] = clf.n_leaves_
        result["scan_counts"] = clf.scan_counts_

        print("PHASE scoring", flush=True)
        score_started = time.perf_counter()
        train_accuracy = clf.score(args.data)
        result["accuracy"] = round(train_accuracy, 4)
        result["train_accuracy"] = round(train_accuracy, 4)
        result["score_seconds"] = round(time.perf_counter() - score_started, 3)

        if args.test_data is not None:
            print("PHASE holdout_scoring", flush=True)
            holdout_started = time.perf_counter()
            result["holdout_accuracy"] = round(clf.score(args.test_data), 4)
            result["holdout_score_seconds"] = round(
                time.perf_counter() - holdout_started, 3
            )

        if args.save_model is not None:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            with args.save_model.open("wb") as handle:
                pickle.dump(clf, handle)
            result["model"] = str(args.save_model)
    else:
        if args.model is None:
            parser.error("--model is required for predict mode")
        with args.model.open("rb") as handle:
            clf = pickle.load(handle)
        print("PHASE predicting", flush=True)
        destination = args.output or Path("/tmp/pdt_predictions.parquet")
        prediction_started = time.perf_counter()
        clf.predict(args.data).select(["prediction"]).sink_parquet(destination)
        result["predict_seconds"] = round(time.perf_counter() - prediction_started, 3)
        result["output"] = str(destination)

    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result["peak_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1
    )
    print("PHASE done", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
