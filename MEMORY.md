# Memory and scan scaling: 1 GB → 8 GB database

This document answers two questions:

1. Does the RAM required by this implementation grow as the database grows?
2. Does the number of full-database scans during training grow as the database
   grows?

The short answers:

* **RAM does not grow with row count / total bytes for a single-file dataset.**
  A 1 GB parquet file and an 8 GB parquet file both peak at roughly 310–350 MB
  with the tuned configuration.  The working set is dominated by Python +
  Polars and the current tree's contingency table, not by the number of rows.
* **The number of scans does not grow with row count when the tree depth is
  capped.**  The measured count was exactly 4 scans (1 root class scan + 3
  levels) at 1, 2, 4 and 8 GB.
* **Total I/O still grows linearly with database size**, because every level
  scans the whole feature+target projection (the root class scan reads only the
  target column).  Time therefore grows linearly for a fixed tree.
  If the tree gets deeper, each extra level adds one more full pass.  That is
  the real scaling risk, not per-node scans.
* **A 100 MB cap is not a meaningful budget for this stack at the 1 GB
  scale.**  Even a 103 MB dataset peaks at ~241 MB; only very small datasets
  (10k rows here, root-only) stayed under 100 MB.

## Test setup

* Base data: 8,000,000 rows × 8 low-cardinality string features + target,
  written without compression or dictionary encoding (1.005 GB, 10k-row
  groups).
* Larger databases: the base file duplicated 2×, 4× and 8× into **single**
  parquet files (2.01, 4.02, 8.04 GB).  A separate multi-file 8 GB case used
  eight 1 GB files to expose per-file overhead.
* Model settings: `target="y"`, `max_depth=3`, `min_samples_split=100_000`,
  `min_samples_leaf=50_000`, `min_gain=1e-6`, `prune=False`.
* Runtime: one Polars thread (`POLARS_MAX_THREADS=1`), Python 3.12, Polars
  2.1.0, NumPy 2.5.5, Apple Silicon, 64 GB RAM, macOS.
* Limit mechanism: this host has no cgroups and macOS refuses
  `RLIMIT_AS`/`RLIMIT_DATA` changes, so `tools/ram_guard.py` samples child RSS
  every 2 ms and kills above the limit.  `ru_maxrss` is reported as an
  independent check.  This is a simulation, not kernel enforcement.

## 100 MB cap: fails

| Phase | Settings | Outcome | Peak RSS |
| --- | --- | --- | ---: |
| Target-only aggregation | 1 thread, 10k row groups | killed | 125.8 MB |
| Training, 1 GB | 1 thread, 10k row groups | killed | 138.0 MB |
| Inference, 1 GB | 1 thread, 10k row groups | killed | 165.2 MB |
| Training, 1 GB | default threads, 500k row groups | killed | 256.8 MB |

The target-only `group_by(target).len()` already exceeds 100 MB.  The minimal
package import is ~64.5 MB, leaving <36 MB for Polars' reader and any data.
`POLARS_MAX_THREADS=1` and smaller row groups reduce memory but cannot get under
100 MB for this workload.

A 10k-row file can train under 100 MB (75 MB peak), but only because it becomes
a single root leaf.  A 100k-row file already exceeds the cap during the first
level scan, and a 103 MB dataset peaks at ~241 MB.  A 100 MB cap is therefore
below the practical floor for anything beyond tiny, root-only datasets.

## Fixed 512 MB cap, database 1 GB → 8 GB

Single-file datasets, one Polars thread, 512 MB cap:

| Database | Rows | Peak RSS | Fit time | Score time | Training scans | Tree |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1.005 GB | 8M | 311.9 MB | 4.61 s | 2.31 s | 4 | depth 3, 90 leaves |
| 2.01 GB | 16M | 350.7 MB | 9.57 s | 5.28 s | 4 | depth 3, 220 leaves |
| 4.02 GB | 32M | 325.1 MB | 19.34 s | 10.60 s | 4 | depth 3, 220 leaves |
| 8.04 GB | 64M | 313.0 MB | 37.84 s | 20.64 s | 4 | depth 3, 220 leaves |

Peak RSS varies by ~±12% due to allocator/measurement noise, but has **no
upward trend** from 1 GB to 8 GB.  The 8 GB database ran under a 512 MB cap:
about 1/16 of the database size at the cap, 1/26 at the observed peak.

The leaf count increases from 90 to 220 between 1 GB and 2 GB because
`min_samples_leaf=50_000` is an absolute threshold: with twice the rows, more
categories qualify for their own child instead of being merged into `other`.
This is a real data-size effect on tree shape, yet the memory stayed flat.

Fit time doubles as the database doubles (4.6 → 9.6 → 19.3 → 37.8 s), and score
time does the same.  That is linear-in-bytes behaviour, not constant-time
streaming.

## Scans and total I/O

The implementation now exposes training scan counts:

```python
clf.scan_counts_
# {'root': 1, 'level': 3, 'validation': 0, 'total': 4}
```

For a tree of depth `d` without pruning:

* 1 full target-column scan to discover classes and root counts,
* `d` full scans, one per level,
* 0 validation scans,
* total `1 + d`.

With `prune=True` and `eval_path`, add one validation scan.  If the root itself
is a leaf, there are no level scans.  Scoring performs one additional pass over
the projected feature+target columns; prediction performs one pass per call.

Crucially, the number of scans per level is **one regardless of the number of
nodes at that level**.  A tree with 10 leaves and a tree with 10,000 leaves at
the same depth both need one scan for that level.  The cost of wider trees
appears in routing, the contingency table and memory, not in extra scans.

Each level scan reads the **whole feature+target projection**; the root scan
reads only the target column.  For the 8 GB case:

* training: 3 level scans × 8 GB plus a target-only root scan (~0.9 GB)
  ≈ 25 GB read, 37.8 s (~0.66 GB/s effective),
* scoring: 8 GB read, 20.6 s (~0.39 GB/s including expression evaluation).

Extrapolating the measured 8-feature, depth-3 workload to 20 TB (2560×):

* 3 feature scans + 1 target scan ≈ 62 TB read, about **27 hours** training,
* scoring another 20 TB, about **15 hours**,
* roughly **42 hours** total.

That is a batch job, not an interactive one.  If the tree needs depth 10 instead
of 3, training needs 10 feature passes plus the target scan instead of 3 + 1:
~74 hours for the same 8-feature data.  Depth 20 → ~141 hours.  These numbers
are linear extrapolations, not benchmarks at 20 TB.

Feature count multiplies the per-row work.  A 100-feature dataset at 20 TB has
roughly 12.5× the per-row CPU work of this 8-feature benchmark, even before
considering higher cardinality and a larger tree.  That is where 20 TB becomes
genuinely impractical.

## Classification accuracy

The memory runs above also report accuracy, but that accuracy was measured on
the training rows.  A proper held-out check was added as well:

* Split the 1 GB base file randomly 80/20 into train (0.804 GB, 6.4M rows) and
  test (0.201 GB, 1.6M rows).
* Train on train only, then score train and test.

| Metric | Value |
| --- | ---: |
| Training accuracy | 0.7387 |
| Held-out accuracy | 0.7379 |
| Train–test gap | 0.0008 |
| Tree | depth 3, 73 leaves |
| Peak RSS | 322.2 MB |
| Scans | 4 |

The tiny gap shows the tree is not memorising the training rows.  The absolute
value is modest because this synthetic generator is noisy; its Bayes-optimal
accuracy is about 0.797.  The larger 2–8 GB runs reached 0.7975 training
accuracy, i.e. the Bayes ceiling, once enough rows made more categories eligible
under `min_samples_leaf=50_000`.  The 1 GB model underfits relative to that
ceiling, which is why its accuracy is lower; this is a data/threshold effect,
not a held-out overfitting effect.

The pytest suite also contains an independent train/test generalization test on
a noisier synthetic concept; it asserts test accuracy above 0.85 and a small
train–test gap.  Existing tests additionally assert exact accuracies (0.625,
0.9, 1.0, etc.) for known small trees and pruning outcomes.

## File-count is a separate memory axis

Scanning one 8 GB file peaked at 313 MB.  Scanning eight 1 GB files containing
the same rows peaked at 397 MB, about +84 MB, or ~10 MB
per extra file.  The current implementation hands the whole source list to one
Polars query, so per-file reader state is held simultaneously.

This matters for a 20 TB deployment split into thousands of small files.  The
current code does not automatically batch files.  For low memory:

* prefer a smaller number of larger parquet files (large files, small row
  groups), or
* split the database into subdirectories and process/batch them outside this
  API in a future version.

Row-group granularity within a file did not show per-row-group memory growth;
10k-row groups kept memory low and also reduced the peak by ~20% versus 500k-row
groups in the single-file tests.

## Why the default run used 2.4 GB

The level scan unpivots all eight feature columns into a long
`(node, feature, value, target)` relation.  For 8M rows that is 64M logical
rows.  Polars streams that relation, but with the default multi-threaded pool
it keeps large per-thread read/decode/aggregate buffers.  Measured effects:

* `POLARS_MAX_THREADS=1` was the largest lever (2.4 GB → ~434 MB for one 1 GB
  file with 500k-row groups; ~313–347 MB with 10k-row groups).
* Smaller parquet row groups reduced memory further.
* `scan_parquet(low_memory=True)` alone did not materially help.
* The final class-count table for this tree is small; intermediate scan and
  thread buffers dominate at low cardinality.

High-cardinality features and unbounded tree growth can still make the
contingency table large.  The flat scaling above is for a low-cardinality
8-feature dataset, not a universal bound.

## Recommended operating point

For a 1–8 GB database on a Python + Polars stack:

* Set `POLARS_MAX_THREADS=1` **before launching Python** (the thread pool is
  initialised at import time).
* Write large parquet files with 10k–50k row groups; avoid thousands of tiny
  files for this version.
* Budget ~512 MB RSS for training and ~256–384 MB for inference.
* Cap tree depth; depth is the number of full-database scans.
* Track `clf.scan_counts_` and compute total I/O as `(1 + depth + prune) ×
  database_size`.
* Do not set a 100 MB target for this stack.  That needs a much smaller native
  runtime, a page-level reader and a custom streaming tree builder.

For 20 TB, memory is not the first wall at this configuration; **time and
feature count are**.  Use feature selection, low cardinality, a shallow
`max_depth`, and a batched file strategy.

## Reproduce

Generate the base dataset (one 1 GB file):

```bash
python tools/make_large_dataset.py \
    --rows 8000000 --path /tmp/pdt_1gb --row-group-size 10000
```

Run under the 100 MB guard (expected to fail):

```bash
POLARS_MAX_THREADS=1 python tools/ram_guard.py --limit-mb 100 -- \
    python -u tools/memory_experiment.py --mode train --data /tmp/pdt_1gb
```

Run under 512 MB and save a model:

```bash
POLARS_MAX_THREADS=1 python tools/ram_guard.py --limit-mb 512 -- \
    python -u tools/memory_experiment.py --mode train --data /tmp/pdt_1gb \
    --save-model /tmp/pdt_model.pkl
```

Inference under 256 MB:

```bash
POLARS_MAX_THREADS=1 python tools/ram_guard.py --limit-mb 256 -- \
    python -u tools/memory_experiment.py --mode predict --data /tmp/pdt_1gb \
    --model /tmp/pdt_model.pkl
```

The JSON printed by `memory_experiment.py` includes `scan_counts`, fit/score
times, tree shape and peak RSS.  `tools/ram_guard.py` prints both the sampled
peak and the child's `ru_maxrss`.
