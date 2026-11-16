# polars-decision-tree

An out-of-core, level-wise **ID3 / C4.5 decision tree** implemented with
[Polars](https://pola.rs/).  It is designed for datasets that do not fit in
memory: training never materialises raw rows in Python, and each tree level is
evaluated with **one scan of the parquet source**, independent of how many
nodes are at that level.

> **Measured memory reality check:** a 1 GB / 8M-row dataset does **not** fit a
> 100 MB total-RSS cap with Python + Polars.  With `POLARS_MAX_THREADS=1` and
> 10k-row parquet groups it trains at ~310–350 MB and predicts at ~230 MB.  An
> 8 GB single-file database also trains at ~313 MB using the same 4 scans; peak
> RAM stays flat while fit time grows linearly.  See [MEMORY.md](MEMORY.md).

```python
from polars_decision_tree import PolarsID3Classifier

clf = PolarsID3Classifier(
    target="label",
    criterion="gain_ratio",
    max_depth=8,
    min_samples_leaf=1_000,
)
clf.fit("data/**/*.parquet", eval_path="validation/**/*.parquet")

print("accuracy:", clf.score("data/**/*.parquet"))
clf.predict("data/**/*.parquet").sink_parquet("predictions.parquet")
clf.export_rules("rules.txt")
```

---

## Why this implementation

The starting point for this project was a level-wise ID3 sketch.  The sketch
had several correctness and scaling defects that this version fixes:

| Reference sketch | This implementation |
| --- | --- |
| Root was created with `is_leaf=True`, then skipped by the level loop, so it never split. | Nodes are explicitly finalised as leaves only after a split search or by pre-pruning. |
| One full source scan **per feature** per level. | One source scan **per level**; all features are unpivoted into one contingency-table aggregation. |
| Classes inferred from the first 10 000 target values. | Classes come from a complete target-column aggregation, so rare classes cannot cause a later `KeyError`. |
| `min_samples_split` was unused. | Enforced before every split. |
| Categories below `min_samples_leaf` were silently dropped from gain and from children. | Rare categories are grouped into an `other` child; a split is rejected if that group is still too small. |
| A `.otherwise()` chain broke for features with more than two categories. | One chained `when → then → … → otherwise` expression supports any number of categories. |
| Pruning issued several validation scans **per node**. | Reduced-error pruning uses one validation scan and a bottom-up dynamic program. |
| Prediction used the same broken `.otherwise()` chain for features with more than two categories. | Prediction compiles one `when` chain and casts each feature exactly once. |

## Installation

```bash
python -m pip install -e .
# development / tests
python -m pip install -e ".[dev]"
```

Requirements: Python ≥ 3.10, Polars ≥ 1.30 (including Polars 2.x), NumPy.
The optional `dev` extra adds `pyarrow` (tests/large-dataset tool),
`psutil` (RSS guard) and `ruff`.

## Data format

Training and prediction data are read with `pl.scan_parquet`, so the source can
be:

* one parquet file,
* a directory (expanded to `**/*.parquet`),
* a glob,
* or a list of files/directories.

The target may be a string, integer, boolean, categorical, date-like scalar,
etc.  Null target labels are rejected.  Features may be any scalar type; they
are cast to strings, so numeric features are treated as categorical values.
Null features become the reserved category `"__pdt_missing__"` (a real
feature value equal to that string is indistinguishable from null).  Bin
continuous features before fitting if they have high cardinality.

## Algorithm

### Level-wise growth

At depth `d` the builder holds a frontier of nodes that may be split.  A single
lazy query:

1. scans the parquet source,
2. casts/normalises feature columns,
3. routes every row to its current node with a compiled `when/then` router,
4. filters to the active frontier,
5. unpivots all features into `(node, feature, value)` form,
6. aggregates `(node, feature, value, class) -> count`.

Only this compact count table is collected.  The builder evaluates every
feature for every node in the frontier, chooses the best split, and creates the
next frontier.  The next level repeats the process.  The number of scans is
therefore `1 + tree_depth + (1 if pruning is used)` (the first scan discovers
the classes and root distribution), not `depth × features` and not a function
of the number of nodes.

### Split criterion

* `gain`: information gain, `H(parent) − Σ (n_i/n) H(child_i)`.
* `gain_ratio`: C4.5 gain divided by split information
  `−Σ (n_i/n) log2(n_i/n)`.

Entropy is in bits.  Ties are broken by feature order, so pass `features=[...]`
in a deterministic order when reproducibility matters.

### Missing values and rare categories

Null feature values become an ordinary category.  During split selection:

* categories with at least `min_samples_leaf` rows become normal children;
* smaller categories are merged into one `other` child, if that child also has
  at least `min_samples_leaf` rows;
* otherwise the feature cannot produce a valid split at that node and is
  skipped.

At prediction, an unseen category goes to the `other` child when one exists.
If there is no `other` child it falls back to the majority class stored at the
internal node, matching the exported fallback rules.

### Pre-pruning

A node becomes a leaf without another scan when:

* it is pure,
* it has fewer than `min_samples_split` rows, or
* it cannot produce two children each with `min_samples_leaf` rows,
* or the maximum depth has been reached.

`min_gain` is applied to every candidate split.

### Reduced-error pruning

When `prune=True` and `eval_path` is supplied, one validation scan routes each
row to its terminal node.  Class counts are propagated bottom-up.  At every
internal node the validation accuracy of the current subtree is compared with
the accuracy of replacing the node by its training-majority leaf; ties are
pruned.  Because decisions are made bottom-up, a parent's comparison already
reflects pruned descendants.  The node's prediction and training counts are not
changed by pruning, except that `gain`/`score` are reset to zero.

Validation rows whose target is null are ignored for pruning.  Validation
labels that never appeared in training count as errors.

## API

### `PolarsID3Classifier`

```python
PolarsID3Classifier(
    target="Class",
    criterion="gain_ratio",   # "gain" | "gain_ratio"
    max_depth=None,           # number of splits from the root; 0 => one leaf
    min_samples_split=2,
    min_samples_leaf=2,
    min_gain=1e-9,
    prune=True,
)
```

* `fit(data_path, features=None, eval_path=None)` — build the tree.  `features`
  defaults to every non-target column.  `eval_path` enables pruning.
* `predict(data_path) -> pl.LazyFrame` — lazy predictions; execute with
  `.collect()`, `.sink_parquet()`, etc.  The output contains the source columns
  plus a `prediction` column.  The operation streams and does not collect row
  data in Python.
* `score(data_path) -> float` — streaming accuracy.
* `export_rules(path) -> Path` — write human-readable rules, including explicit
  fallback rules for unseen categories.

Fitted attributes: `classes_`, `feature_names_`, `n_features_in_`,
`target_dtype_`, `root_`, `depth_`, `n_leaves_`, and `scan_counts_` (training
source scans split into `root`, `level`, `validation`, `total`).

### `Node`

`Node` exposes `is_leaf`, `feature`, `children`, `other_child`, `prediction`,
`proba`, `n_samples`, `gain`, `score`, `depth`, and `id`.  `children` maps
observed category strings to child nodes; `other_child` is the rare/unseen
category branch.

## Scale and performance notes

This is an exact multiway ID3 implementation.  A single level performs
`O(rows × features)` work because every feature value contributes a count for
every row, and the streaming group-by holds one state entry per
`(node, feature, value, class)` group.  That is fundamentally different from a
binary numeric tree, which can examine one threshold per feature.

Consequences:

* **20 TB is I/O- and CPU-bound.**  Exact ID3 over 20 TB requires an enormous
  amount of work regardless of RAM.  The implementation is out-of-core, but
  exact training over 20 TB is not a short job.  Use feature selection, low
  cardinality, `min_samples_leaf`, `max_depth`, and small row groups.  Keep
  parquet files reasonably large: the current implementation passes the whole
  source list to one Polars query, and many tiny files add per-file memory.
* **At low feature cardinality, RAM is dominated by Polars' scan/unpivot
  buffers, not raw rows.**  On the 1 GB benchmark, one thread plus small row
  groups lowered the peak from 2.4 GB to ~312–351 MB.  High-cardinality features
  and many leaves can make the contingency table itself the dominant cost;
  bin numeric columns first and avoid ID-like columns.
* **Polars streaming is used for every aggregate.**  No Pandas conversion and
  no per-row Python loop over the data.  The final count table is converted to
  per-node NumPy count arrays.
* **Scan budget = 1 + tree depth (+1 for pruning).**  The root scan discovers
  classes and reads only the target column; each level adds one full scan of
  the feature+target projection.  `max_depth` is therefore an I/O knob: a
  depth-3 tree needs 4 logical scans at any row count, while an unbounded,
  deeper tree adds a full pass per level.  `clf.scan_counts_` reports the
  actual training scans.
* **Prediction is lazy** and can be sunk directly to parquet, so predictions
  also need not fit in memory.
* **Pruning is optional** and adds one validation scan (except when the tree is
  already a single leaf); without `eval_path`, `prune=True` has no effect.
* **Keep the training source immutable during a fit.**  Successive levels must
  see the same rows; otherwise the per-node counts stored from one level will
  disagree with the next level's scan (the implementation raises a
  `RuntimeError` rather than silently producing a corrupt tree).

A local sanity benchmark (1 000 000 rows × 8 categorical features, 4–18
categories each, max depth 6) fits in under two seconds on an Apple-silicon
laptop; that number is illustrative, not a promise for wide or very
high-cardinality data.  Reproduce it with:

```bash
python examples/benchmark.py --rows 1000000 --features 8
```

`examples/benchmark.py` measures generation as well as training; see its
docstring for details.

### Measured memory

The repository includes a reproducible memory experiment and an RSS guard:

```bash
python tools/make_large_dataset.py \
    --rows 8000000 --path /tmp/pdt_1gb --row-group-size 10000

POLARS_MAX_THREADS=1 python tools/ram_guard.py --limit-mb 100 -- \
    python -u tools/memory_experiment.py --mode train --data /tmp/pdt_1gb
```

On a 1 GB dataset (8M rows, 8 low-cardinality categorical features):

| Run | Cap | Peak RSS | Result |
| --- | ---: | ---: | --- |
| Training, default threads/row groups | none | 2424.7 MB | completed |
| Training, 1 thread, 10k row groups | 100 MB | 138.0 MB | killed |
| Training, 1 thread, 10k row groups | 384 MB | 330.5 MB | completed |
| Training, 8 GB single file, 1 thread | 512 MB | 313.0 MB | completed |
| Training, 8 GB as eight 1 GB files | 512 MB | 396.8 MB | completed |
| Inference, 1 thread, 10k row groups | 100 MB | 165.2 MB | killed |
| Inference, 1 thread, 10k row groups | 256 MB | 229.7 MB | completed |

A target-only aggregation alone peaks at ~126 MB with one thread, which puts a
100 MB cap below the runtime floor.  For single-file data, peak RSS stayed
roughly flat from 1 GB to 8 GB (312–351 MB) while fit time grew 8× (4.6 s →
37.8 s).  A proper 80/20 holdout on the 1 GB data scored 0.7379 versus 0.7387
on the training split (gap 0.0008).  Training always reported 4 logical scans: 1 target-only root scan + 3
level scans.  The scan count is independent of row count and node count, but
each level scan reads the whole feature+target projection, so I/O and time grow
linearly with database size (and one extra full pass per extra tree level).
Full methodology, scan/I/O extrapolations and caveats are in
[MEMORY.md](MEMORY.md).

## Development

```bash
python -m pytest -q
ruff check src tests examples tools
ruff format --check src tests examples tools
```

The test suite covers:

* perfect and max-depth-limited trees,
* gain vs. gain ratio choosing different features,
* full-scan class discovery,
* missing values and unseen categories,
* rare-category grouping and `min_samples_leaf` rejection,
* numeric, categorical, and boolean targets,
* source directories, globs, and file lists,
* reduced-error pruning (including nested and supported/unsupported splits),
* lazy prediction and `sink_parquet`,
* independent train/test holdout accuracy (not only re-scoring training rows),
* rule export and input validation,
* one level scan per depth (instrumented test), with `scan_counts_`.

## License

This project is licensed under the **GNU General Public License v3.0**
(GPL-3.0-only).  See [LICENSE](LICENSE) for the full text.

GPLv3 is a strong copyleft license: if you distribute modified versions, you
must make the corresponding source available under the same license.
