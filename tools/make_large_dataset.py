#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Create a roughly target-sized categorical parquet dataset for memory tests.

The file is deliberately written without dictionary encoding or compression so
its on-disk size is close to the selected-column bytes the model must read.

    python tools/make_large_dataset.py --rows 8000000 --path /tmp/pdt_1gb

Features are low-cardinality strings, which keeps the tree's contingency table
small and lets the experiment focus on scan/streaming memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8_000_000)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--path", type=Path, default=Path("/tmp/pdt_1gb"))
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--row-group-size", type=int, default=250_000)
    args = parser.parse_args()

    args.path.mkdir(parents=True, exist_ok=True)
    destination = args.path / "data.parquet"
    schema = pa.schema(
        [(f"f{i}", pa.string()) for i in range(args.features)] + [("y", pa.string())]
    )
    rng = np.random.default_rng(42)

    with pq.ParquetWriter(
        destination,
        schema,
        compression="NONE",
        use_dictionary=False,
        write_statistics=True,
    ) as writer:
        written = 0
        while written < args.rows:
            n = min(args.chunk_rows, args.rows - written)
            codes = np.empty((n, args.features), dtype=np.int16)
            arrays = []
            for i in range(args.features):
                cardinality = 4 + (i % 6) * 3
                codes[:, i] = rng.integers(0, cardinality, size=n)
                values = np.char.mod("category_%03d", codes[:, i]).tolist()
                arrays.append(pa.array(values, type=pa.string()))

            signal = (
                (codes[:, 0] % 3 == 0).astype(np.int8)
                + (codes[:, 1] % 4 == 0).astype(np.int8)
                + (codes[:, 2] % 2 == 0).astype(np.int8)
            )
            noise = rng.integers(0, 3, size=n)
            target = np.where(signal + noise >= 2, "yes", "no")
            arrays.append(pa.array(target.tolist(), type=pa.string()))

            writer.write_table(
                pa.Table.from_arrays(arrays, schema=schema),
                row_group_size=args.row_group_size,
            )
            written += n
            if written % (args.chunk_rows * 8) == 0:
                print(f"wrote {written} rows", flush=True)

    size_gb = destination.stat().st_size / 1024**3
    print(f"rows={args.rows} path={destination} size_gb={size_gb:.3f}")


if __name__ == "__main__":
    main()
