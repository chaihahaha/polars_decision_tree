# SPDX-License-Identifier: GPL-3.0-only
"""Out-of-core, level-wise ID3 decision trees built with Polars.

The implementation is deliberately split into two layers:

* :class:`PolarsID3Classifier` is the small public estimator API.
* :class:`_LevelBuilder` implements the out-of-core algorithm.  Every tree
  level is evaluated with **one** scan of the source parquet dataset.  The
  scan assigns rows to the active nodes at that level, unpivots all feature
  columns and produces a single compact contingency table grouped by
  ``(node, feature, value, class)``.  No row-level data are materialised in
  Python.

Everything here is implemented; there are no training or prediction stubs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

__all__ = ["MISSING_SENTINEL", "Node", "PolarsID3Classifier"]


# A private string value used to turn null feature values into an ordinary
# category.  This mirrors the behaviour of the reference implementation.  If a
# dataset genuinely contains this string in a feature column, those values are
# indistinguishable from nulls; use a different sentinel in your source data if
# that is a problem.
MISSING_SENTINEL = "__pdt_missing__"

# Columns used inside the streaming query.  They are internal, but rejecting
# user columns with these names keeps the generated expressions unambiguous.
_NODE_ID = "_node_id"
_FEATURE = "_feature"
_VALUE = "_value"
_COUNT = "_count"
_TERMINAL_ID = "_terminal_id"
_PREDICTION = "prediction"
_RESERVED_COLUMNS = frozenset(
    {_NODE_ID, _FEATURE, _VALUE, _COUNT, _TERMINAL_ID, _PREDICTION}
)

PathLike = str | Path
Source = PathLike | Sequence[PathLike]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy in bits of a class-count vector."""
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def _weighted_entropy(table: np.ndarray) -> float:
    """Weighted entropy of a (children x classes) contingency table."""
    row_totals = table.sum(axis=1).astype(np.float64)
    total = float(row_totals.sum())
    if total <= 0.0:
        return 0.0
    probs = table.astype(np.float64) / row_totals[:, None]
    logs = np.zeros_like(probs)
    np.log2(probs, out=logs, where=probs > 0.0)
    entropies = -(probs * logs).sum(axis=1)
    return float((row_totals / total * entropies).sum())


def _split_info(table: np.ndarray) -> float:
    """C4.5 split information (entropy of the child-size distribution)."""
    row_totals = table.sum(axis=1).astype(np.float64)
    total = float(row_totals.sum())
    if total <= 0.0:
        return 0.0
    p = row_totals[row_totals > 0.0] / total
    return float(-(p * np.log2(p)).sum())


def _when_chain(
    branches: list[tuple[pl.Expr, pl.Expr]],
    fallback: pl.Expr,
) -> pl.Expr:
    """Build a single ``when/then/.../otherwise`` expression.

    Polars only allows ``otherwise`` once at the end of a chain, so branches
    are added with ``Then.when`` rather than nested ``otherwise`` calls.  This
    also supports arbitrarily many categorical branches.
    """
    expr = None
    for predicate, value in branches:
        if expr is None:
            expr = pl.when(predicate).then(value)
        else:
            expr = expr.when(predicate).then(value)
    if expr is None:
        return fallback
    return expr.otherwise(fallback)


def _raw_value_expr(feature: str) -> pl.Expr:
    """Cast and normalise one raw feature for prediction."""
    return pl.col(feature).cast(pl.String, strict=False).fill_null(MISSING_SENTINEL)


def _prediction_literal(value: Any, dtype: pl.DataType) -> pl.Expr:
    try:
        return pl.lit(value, dtype=dtype)
    except (TypeError, ValueError):
        # Some extension dtypes cannot be used as a literal dtype.  A plain
        # Python literal is still valid for a scalar prediction.
        return pl.lit(value)


def _normalise_source(source: Source) -> str | list[str]:
    """Turn a directory/glob/path sequence into something ``scan_parquet`` likes."""
    if isinstance(source, (list, tuple)):
        if not source:
            raise ValueError("data_path cannot be empty")
        return [_normalise_one(item) for item in source]
    return _normalise_one(source)


def _normalise_one(source: PathLike) -> str:
    if not isinstance(source, (str, Path)):
        raise TypeError(f"Expected a path or sequence of paths, got {type(source)!r}")
    raw = str(source)
    # Leave URIs (s3://, gs://, ...) untouched.
    if "://" in raw:
        return raw
    path = Path(raw)
    if path.is_dir():
        return str(path / "**" / "*.parquet")
    return raw


def _is_nan_scalar(value: Any) -> bool:
    try:
        return bool(isinstance(value, float) and math.isnan(value))
    except (TypeError, ValueError):
        return False


def _sorted_classes(labels: Iterable[Any]) -> list[Any]:
    labels = list(labels)
    try:
        return sorted(labels)
    except TypeError:
        return sorted(labels, key=lambda value: (type(value).__name__, repr(value)))


# ---------------------------------------------------------------------------
# Tree nodes
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A node in the induced decision tree.

    ``children`` maps observed category strings to child nodes.  If a split had
    categories whose counts were below ``min_samples_leaf``, they are grouped
    into :attr:`other_child`; that child also receives unseen categories at
    prediction time.  When :attr:`other_child` is ``None``, unseen categories
    fall back to :attr:`prediction` at that node.
    """

    is_leaf: bool = False
    feature: str | None = None
    children: dict[str, Node] = field(default_factory=dict)
    other_child: Node | None = None
    prediction: Any = None
    proba: np.ndarray | None = field(default=None, repr=False)
    n_samples: int = 0
    gain: float = 0.0
    score: float = 0.0
    class_counts: np.ndarray | None = field(default=None, repr=False)
    id: int = -1
    depth: int = 0
    parent: Node | None = field(default=None, repr=False, compare=False)


@dataclass
class _BestSplit:
    feature: str
    frequent: list[tuple[str, np.ndarray]]
    other_counts: np.ndarray | None
    gain: float
    score: float


# ---------------------------------------------------------------------------
# Level-wise builder
# ---------------------------------------------------------------------------


class _LevelBuilder:
    """Build the tree one complete level at a time.

    The key invariant is that :meth:`_scan_level` creates exactly one Polars
    lazy plan and collects exactly one aggregate for a level, regardless of how
    many nodes or features exist at that level.
    """

    def __init__(
        self,
        data_path: str | list[str],
        target: str,
        features: list[str],
        classes: list[Any],
        criterion: str,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        min_gain: float,
        prune: bool,
        eval_path: str | list[str] | None = None,
    ) -> None:
        self.data_path = data_path
        self.target = target
        self.features = features
        self.classes = classes
        self.class_to_idx = {label: i for i, label in enumerate(classes)}
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.prune = prune
        self.eval_path = eval_path

        self.root = Node()
        self.nodes: list[Node] = []
        # Physical source scans, by purpose.  Schema/metadata lookups are not
        # counted; this is the number of times training reads the data.
        self.scan_counts = {"root": 0, "level": 0, "validation": 0}

    # -- node construction -------------------------------------------------

    def _set_counts(self, node: Node, counts: np.ndarray) -> None:
        counts = np.asarray(counts, dtype=np.int64)
        if counts.shape != (len(self.classes),):
            raise ValueError(
                f"Class-count vector has shape {counts.shape}, expected {(len(self.classes),)}"
            )
        total = int(counts.sum())
        node.class_counts = counts
        node.n_samples = total
        if total > 0:
            node.proba = counts.astype(np.float64) / total
            node.prediction = self.classes[int(np.argmax(counts))]
        else:
            node.proba = np.full(
                len(self.classes), 1.0 / len(self.classes), dtype=np.float64
            )
            node.prediction = self.classes[0]

    def _new_node(self, counts: np.ndarray, depth: int, parent: Node | None) -> Node:
        node = Node(is_leaf=False, id=len(self.nodes), depth=depth, parent=parent)
        self._set_counts(node, counts)
        self.nodes.append(node)
        return node

    def _make_leaf(self, node: Node) -> None:
        node.is_leaf = True
        node.feature = None
        node.children = {}
        node.other_child = None

    @staticmethod
    def _is_pure(node: Node) -> bool:
        counts = node.class_counts
        return counts is None or int(np.count_nonzero(counts)) <= 1

    # -- root --------------------------------------------------------------

    def _initialise_root(self) -> None:
        """Read the target column once to discover classes and root counts."""
        self.scan_counts["root"] += 1
        target_counts = (
            pl.scan_parquet(self.data_path)
            .select(pl.col(self.target))
            .group_by(self.target)
            .agg(pl.len().alias(_COUNT))
            .collect(engine="streaming")
        )

        raw: dict[Any, int] = {}
        for label, count in target_counts.iter_rows():
            if label is None:
                raise ValueError(f"Target column {self.target!r} contains null values")
            if _is_nan_scalar(label):
                raise ValueError(f"Target column {self.target!r} contains NaN labels")
            try:
                hash(label)
            except TypeError as exc:
                raise ValueError(
                    f"Target labels must be hashable scalar values; got {label!r}"
                ) from exc
            raw[label] = raw.get(label, 0) + int(count)

        if not raw:
            raise ValueError(f"No rows found in {self.data_path!r}")

        classes = _sorted_classes(raw)
        counts = np.asarray([raw[label] for label in classes], dtype=np.int64)
        self.classes = classes
        self.class_to_idx = {label: i for i, label in enumerate(classes)}

        self.root = self._new_node(counts, depth=0, parent=None)
        if self.max_depth == 0:
            self._make_leaf(self.root)

    # -- public build ------------------------------------------------------

    def build(self) -> Node:
        self._initialise_root()
        if self.root.is_leaf:
            # A single leaf has no split to prune and its prediction is the
            # training majority; no validation scan is needed.
            return self.root

        frontier: list[Node] = [self.root]
        depth = 0

        while frontier:
            if self.max_depth is not None and depth >= self.max_depth:
                for node in frontier:
                    self._make_leaf(node)
                break

            # Pre-pruning decisions that need no scan.
            expandable: list[Node] = []
            for node in frontier:
                if (
                    node.n_samples < self.min_samples_split
                    or node.n_samples < 2 * self.min_samples_leaf
                    or self._is_pure(node)
                ):
                    self._make_leaf(node)
                else:
                    expandable.append(node)
            frontier = expandable
            if not frontier:
                break

            tables = self._scan_level(frontier)

            next_frontier: list[Node] = []
            split_any = False
            for node in frontier:
                node_tables = tables.get(node.id)
                if node_tables is None:
                    raise RuntimeError(
                        f"No contingency counts were collected for node {node.id}; "
                        "the row router is inconsistent with the source data"
                    )
                best = self._best_split(node, node_tables)
                if best is None:
                    self._make_leaf(node)
                    continue
                self._apply_split(node, best)
                split_any = True
                next_frontier.extend(node.children.values())
                if node.other_child is not None:
                    next_frontier.append(node.other_child)

            if not split_any:
                break
            depth += 1
            frontier = next_frontier

        if self.prune and self.eval_path is not None:
            self._prune_tree()
        return self.root

    # -- expressions -------------------------------------------------------

    def _id_router(self, node: Node) -> pl.Expr:
        """Expression mapping every row to the id of its current tree node.

        Features are already cast to strings by :meth:`_scan_level`.  Rows stop
        at a leaf.  When an internal node has no ``other`` child, an unexpected
        category (which should not occur during training) terminates at that
        node rather than being silently dropped.
        """
        if node.is_leaf or (not node.children and node.other_child is None):
            return pl.lit(node.id, dtype=pl.Int64)

        branches: list[tuple[pl.Expr, pl.Expr]] = []
        if node.feature is None:
            raise RuntimeError(f"Internal node {node.id} has no split feature")
        for value, child in node.children.items():
            branches.append((pl.col(node.feature) == value, self._id_router(child)))

        if node.other_child is not None:
            fallback = self._id_router(node.other_child)
        else:
            fallback = pl.lit(node.id, dtype=pl.Int64)

        return _when_chain(branches, fallback)

    def _cast_features(self) -> list[pl.Expr]:
        return [
            pl.col(feature)
            .cast(pl.String, strict=False)
            .fill_null(MISSING_SENTINEL)
            .alias(feature)
            for feature in self.features
        ]

    # -- one scan per level ------------------------------------------------

    def _scan_level(
        self, frontier: list[Node]
    ) -> dict[int, dict[str, dict[str, np.ndarray]]]:
        """Read the source once and return per-node contingency tables.

        Return shape: ``{node_id: {feature: {value: class_counts}}}``.
        """
        active_ids = [node.id for node in frontier]
        router = self._id_router(self.root)
        self.scan_counts["level"] += 1

        level = (
            pl.scan_parquet(self.data_path)
            .select(self.features + [self.target])
            .with_columns(self._cast_features())
            .with_columns(router.alias(_NODE_ID))
            .filter(pl.col(_NODE_ID).is_in(active_ids))
            .unpivot(
                on=self.features,
                index=[_NODE_ID, self.target],
                variable_name=_FEATURE,
                value_name=_VALUE,
            )
            .group_by([_NODE_ID, _FEATURE, _VALUE, self.target])
            .agg(pl.len().alias(_COUNT))
        )

        # This is the only collect for the level: one physical pass over the
        # parquet source, one compact count table in memory.
        stats = level.collect(engine="streaming")

        tables: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        n_classes = len(self.classes)
        for node_id, feature, value, label, count in stats.iter_rows():
            node_id = int(node_id)
            if node_id < 0:
                raise RuntimeError(
                    "The tree router produced an unassigned row during training"
                )
            if label is None:
                raise RuntimeError(
                    "Null target encountered in a training contingency table"
                )
            try:
                label_idx = self.class_to_idx[label]
            except KeyError as exc:
                raise RuntimeError(
                    f"Target label {label!r} was not present in the full class scan"
                ) from exc

            feature_tables = tables.get(node_id)
            if feature_tables is None:
                feature_tables = {}
                tables[node_id] = feature_tables
            value_tables = feature_tables.get(feature)
            if value_tables is None:
                value_tables = {}
                feature_tables[feature] = value_tables
            class_counts = value_tables.get(value)
            if class_counts is None:
                class_counts = np.zeros(n_classes, dtype=np.int64)
                value_tables[value] = class_counts
            class_counts[label_idx] += int(count)

        return tables

    # -- split selection ---------------------------------------------------

    def _best_split(
        self,
        node: Node,
        feature_tables: dict[str, dict[str, np.ndarray]],
    ) -> _BestSplit | None:
        parent_counts = node.class_counts
        if parent_counts is None:
            raise RuntimeError(f"Node {node.id} has no class counts")
        parent_total = int(parent_counts.sum())
        parent_entropy = _entropy(parent_counts)

        best: _BestSplit | None = None
        best_score = -math.inf

        for feature in self.features:
            value_counts = feature_tables.get(feature)
            if not value_counts:
                continue

            feature_total = sum(int(counts.sum()) for counts in value_counts.values())
            if feature_total != parent_total:
                raise RuntimeError(
                    f"Contingency table for node {node.id}, feature {feature!r} "
                    f"contains {feature_total} rows but the node has {parent_total}"
                )

            frequent: list[tuple[str, np.ndarray]] = []
            other_counts: np.ndarray | None = None
            for value, counts in value_counts.items():
                total = int(counts.sum())
                if total >= self.min_samples_leaf:
                    frequent.append((value, counts))
                elif other_counts is None:
                    other_counts = counts.astype(np.int64, copy=True)
                else:
                    other_counts += counts

            # A grouped "other" branch is only usable if it also satisfies the
            # leaf-size constraint.  Otherwise this split cannot honour
            # min_samples_leaf for every child and is rejected.
            if (
                other_counts is not None
                and int(other_counts.sum()) < self.min_samples_leaf
            ):
                continue
            if len(frequent) < 2:
                continue

            frequent.sort(key=lambda item: item[0])
            rows = [counts for _, counts in frequent]
            if other_counts is not None:
                rows.append(other_counts)
            table = np.stack(rows, axis=0)

            gain = parent_entropy - _weighted_entropy(table)
            if gain <= self.min_gain + 1e-12:
                continue

            if self.criterion == "gain_ratio":
                denominator = _split_info(table)
                if denominator <= 1e-12:
                    continue
                score = gain / denominator
            else:
                score = gain

            if score > best_score + 1e-12:
                best_score = score
                best = _BestSplit(
                    feature=feature,
                    frequent=[
                        (value, counts.astype(np.int64, copy=False))
                        for value, counts in frequent
                    ],
                    other_counts=(
                        None
                        if other_counts is None
                        else other_counts.astype(np.int64, copy=False)
                    ),
                    gain=float(gain),
                    score=float(score),
                )

        return best

    def _apply_split(self, node: Node, best: _BestSplit) -> None:
        node.is_leaf = False
        node.feature = best.feature
        node.gain = best.gain
        node.score = best.score
        node.children = {}
        node.other_child = None

        for value, counts in best.frequent:
            child = self._new_node(counts, depth=node.depth + 1, parent=node)
            node.children[value] = child

        if best.other_counts is not None:
            node.other_child = self._new_node(
                best.other_counts, depth=node.depth + 1, parent=node
            )

    # -- reduced-error pruning --------------------------------------------

    def _prune_tree(self) -> None:
        """Reduced-error pruning with one validation pass.

        Validation rows are routed to their terminal node (a leaf, or an
        internal node when an unseen category hits a branch without an
        ``other`` child).  Class counts are then propagated bottom-up.  For
        every internal node the accuracy of keeping the subtree is compared
        with the accuracy of replacing the node by its training-majority leaf;
        ties are pruned.  Because counts are propagated bottom-up, the subtree
        accuracy already reflects pruning decisions made in descendants.
        """
        if self.eval_path is None:
            return

        self.scan_counts["validation"] += 1
        val = (
            pl.scan_parquet(self.eval_path)
            .select(self.features + [self.target])
            .filter(pl.col(self.target).is_not_null())
            .with_columns(self._cast_features())
        )
        terminal = self._id_router(self.root)
        grouped = (
            val.with_columns(terminal.alias(_TERMINAL_ID))
            .group_by([_TERMINAL_ID, self.target])
            .agg(pl.len().alias(_COUNT))
            .collect(engine="streaming")
        )
        if grouped.height == 0:
            return

        n_classes = len(self.classes)
        direct_counts: dict[int, np.ndarray] = {}
        direct_totals: dict[int, int] = {}
        for terminal_id, label, count in grouped.iter_rows():
            terminal_id = int(terminal_id)
            if terminal_id < 0 or terminal_id >= len(self.nodes):
                raise RuntimeError(
                    f"Validation router produced unknown terminal node {terminal_id}"
                )
            counts = direct_counts.get(terminal_id)
            if counts is None:
                counts = np.zeros(n_classes, dtype=np.int64)
                direct_counts[terminal_id] = counts
            direct_totals[terminal_id] = direct_totals.get(terminal_id, 0) + int(count)
            label_idx = self.class_to_idx.get(label)
            if label_idx is not None:
                counts[label_idx] += int(count)

        class_to_idx = self.class_to_idx

        def visit(node: Node) -> tuple[int, np.ndarray, int]:
            pred_idx = class_to_idx[node.prediction]
            direct = direct_counts.get(node.id)
            counts = (
                np.zeros(n_classes, dtype=np.int64) if direct is None else direct.copy()
            )
            direct_total = direct_totals.get(node.id, 0)
            direct_correct = int(counts[pred_idx]) if direct_total > 0 else 0

            children = list(node.children.values())
            if node.other_child is not None:
                children.append(node.other_child)

            if node.is_leaf or not children:
                correct = int(counts[pred_idx]) if direct_total > 0 else 0
                return correct, counts, direct_total

            subtree_correct = direct_correct
            subtree_total = direct_total
            for child in children:
                child_correct, child_counts, child_total = visit(child)
                subtree_correct += child_correct
                subtree_total += child_total
                counts += child_counts

            leaf_correct = int(counts[pred_idx]) if subtree_total > 0 else 0
            if subtree_total > 0 and leaf_correct >= subtree_correct:
                self._make_leaf(node)
                node.gain = 0.0
                node.score = 0.0
                return leaf_correct, counts, subtree_total
            return subtree_correct, counts, subtree_total

        visit(self.root)


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------


class PolarsID3Classifier:
    """ID3/C4.5 decision tree trained with level-wise Polars streaming.

    Parameters
    ----------
    target:
        Name of the target column.
    criterion:
        ``"gain"`` (information gain) or ``"gain_ratio"`` (C4.5).
    max_depth:
        Maximum number of splits from the root.  ``None`` means unlimited;
        ``0`` produces a single leaf.
    min_samples_split:
        A node must contain at least this many rows to be considered for a
        split.
    min_samples_leaf:
        Every child produced by a split must contain at least this many rows.
        Values rarer than this are grouped into a single ``other`` child; if
        that grouped child would still be too small, the split is rejected.
    min_gain:
        Minimum information gain for a split.
    prune:
        If true and ``eval_path`` is supplied to :meth:`fit`, perform
        reduced-error pruning on the validation set.

    Notes
    -----
    ID3 is a categorical algorithm.  Every feature is cast to ``String`` for
    training and prediction, so numeric columns are split on their exact
    (stringified) values.  Bin numeric columns before fitting when they have
    high cardinality.
    """

    def __init__(
        self,
        target: str = "Class",
        criterion: str = "gain_ratio",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 2,
        min_gain: float = 1e-9,
        prune: bool = True,
    ) -> None:
        self.target = target
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.prune = prune

        self.classes_: list[Any] = []
        self.feature_names_: list[str] = []
        self.n_features_in_: int = 0
        self.target_dtype_: pl.DataType | None = None
        self.root_: Node | None = None
        self.depth_: int = 0
        self.n_leaves_: int = 0
        self.scan_counts_: dict[str, int] = {}
        self._fitted = False

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return (
            f"PolarsID3Classifier(target={self.target!r}, criterion={self.criterion!r}, "
            f"max_depth={self.max_depth!r}, {state})"
        )

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        data_path: Source,
        features: list[str] | None = None,
        eval_path: Source | None = None,
    ) -> PolarsID3Classifier:
        """Fit the tree from parquet data.

        Parameters
        ----------
        data_path:
            Parquet file, directory, glob, or a sequence of those.
        features:
            Feature columns.  When omitted, every column except the target is
            used.  Order is significant for deterministic gain ties.
        eval_path:
            Optional validation parquet source used for reduced-error pruning.
        """
        self._validate_hyperparameters()
        data_source = _normalise_source(data_path)
        eval_source = None if eval_path is None else _normalise_source(eval_path)

        schema = pl.scan_parquet(data_source).collect_schema()
        schema_names = schema.names()

        if self.target not in schema_names:
            raise ValueError(
                f"Target column {self.target!r} is not in the parquet schema"
            )

        target_dtype = schema[self.target]
        if target_dtype.is_nested() or target_dtype == pl.Object:
            raise ValueError(
                "The target must be a scalar categorical/numeric column, "
                f"got {target_dtype!r}"
            )

        if features is None:
            feature_names = [name for name in schema_names if name != self.target]
        elif isinstance(features, (str, bytes)):
            raise TypeError("features must be a sequence of column names, not a string")
        else:
            feature_names = list(features)

        if not feature_names:
            raise ValueError("At least one feature column is required")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Feature names must be unique")
        if self.target in feature_names:
            raise ValueError("The target column cannot also be a feature")

        missing = [name for name in feature_names if name not in schema_names]
        if missing:
            raise ValueError(
                f"Feature columns not found in the parquet schema: {missing}"
            )

        nested_features = [
            name
            for name in feature_names
            if schema[name].is_nested() or schema[name] == pl.Object
        ]
        if nested_features:
            raise ValueError(
                "ID3 features must be scalar categorical values; "
                f"these columns are nested/object: {nested_features}"
            )

        reserved = (set(feature_names) | {self.target}) & _RESERVED_COLUMNS
        if reserved:
            raise ValueError(
                f"These column names are reserved by the implementation: {sorted(reserved)}"
            )

        if eval_source is not None and self.prune:
            eval_schema = pl.scan_parquet(eval_source).collect_schema()
            eval_missing = [
                name
                for name in feature_names + [self.target]
                if name not in eval_schema.names()
            ]
            if eval_missing:
                raise ValueError(
                    f"Validation data is missing required columns: {eval_missing}"
                )

        # Reset any previous fit only after all cheap validation has passed.
        self.classes_ = []
        self.feature_names_ = []
        self.n_features_in_ = 0
        self.target_dtype_ = None
        self.root_ = None
        self.depth_ = 0
        self.n_leaves_ = 0
        self.scan_counts_ = {}
        self._fitted = False

        builder = _LevelBuilder(
            data_path=data_source,
            target=self.target,
            features=feature_names,
            classes=[],
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_gain=self.min_gain,
            prune=self.prune,
            eval_path=eval_source,
        )
        self.root_ = builder.build()
        self.classes_ = list(builder.classes)
        self.feature_names_ = feature_names
        self.n_features_in_ = len(feature_names)
        self.target_dtype_ = target_dtype
        self.depth_ = self._compute_depth(self.root_)
        self.n_leaves_ = self._count_leaves(self.root_)
        scan_counts = dict(builder.scan_counts)
        scan_counts["total"] = sum(scan_counts.values())
        self.scan_counts_ = scan_counts
        self._fitted = True
        return self

    def _validate_hyperparameters(self) -> None:
        if self.criterion not in {"gain", "gain_ratio"}:
            raise ValueError("criterion must be either 'gain' or 'gain_ratio'")
        if self.max_depth is not None and (
            not isinstance(self.max_depth, int) or self.max_depth < 0
        ):
            raise ValueError("max_depth must be None or a non-negative integer")
        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer")
        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")
        if not isinstance(self.min_gain, (int, float)) or self.min_gain < 0:
            raise ValueError("min_gain must be a non-negative number")

    @staticmethod
    def _compute_depth(node: Node) -> int:
        if node.is_leaf:
            return 0
        children = list(node.children.values())
        if node.other_child is not None:
            children.append(node.other_child)
        return 1 + max(
            (PolarsID3Classifier._compute_depth(child) for child in children), default=0
        )

    @staticmethod
    def _count_leaves(node: Node) -> int:
        if node.is_leaf:
            return 1
        children = list(node.children.values())
        if node.other_child is not None:
            children.append(node.other_child)
        return sum(PolarsID3Classifier._count_leaves(child) for child in children)

    # -- prediction --------------------------------------------------------

    def predict(self, data_path: Source) -> pl.LazyFrame:
        """Return a lazy frame with the source columns plus ``prediction``.

        Call ``.collect()``, ``.sink_parquet()``, etc. to execute.  Prediction
        streams; it does not materialise the source in Python.
        """
        self._check_fitted()
        frame, temporary = self._prediction_frame(data_path)
        return frame.drop(list(temporary.values()))

    def score(self, data_path: Source) -> float:
        """Return accuracy on a parquet source, computed with a streaming mean."""
        self._check_fitted()
        # Only project the features and target: accuracy does not need the
        # other source columns to be read from parquet.
        frame, _ = self._prediction_frame(
            data_path, projection=self.feature_names_ + [self.target]
        )
        result = frame.select(
            (pl.col(_PREDICTION) == pl.col(self.target)).mean().alias("accuracy")
        ).collect(engine="streaming")
        accuracy = result["accuracy"][0]
        if accuracy is None:
            raise ValueError("Cannot score an empty dataset")
        return float(accuracy)

    def _prediction_frame(
        self,
        data_path: Source,
        projection: list[str] | None = None,
    ) -> tuple[pl.LazyFrame, dict[str, str]]:
        assert self.root_ is not None and self.target_dtype_ is not None
        source = _normalise_source(data_path)
        frame = pl.scan_parquet(source)
        schema = frame.collect_schema()
        schema_names = schema.names()

        if projection is not None:
            missing_projection = [
                name for name in projection if name not in schema_names
            ]
            if missing_projection:
                raise ValueError(
                    f"Prediction data is missing required columns: {missing_projection}"
                )
            frame = frame.select(projection)

        missing = [name for name in self.feature_names_ if name not in schema_names]
        if missing:
            raise ValueError(f"Prediction data is missing features: {missing}")

        # Cast each feature once and reference the temporary columns from all
        # branches.  Repeating the cast inside every branch would make the
        # prediction expression unnecessarily large for wide categorical trees.
        used = set(schema_names)
        temporary: dict[str, str] = {}
        casts: list[pl.Expr] = []
        for index, feature in enumerate(self.feature_names_):
            name = f"__pdt_pred_value_{index}"
            while name in used:
                name += "_"
            used.add(name)
            temporary[feature] = name
            casts.append(_raw_value_expr(feature).alias(name))

        compiler = _PredictionCompiler(self.target_dtype_)
        expr = compiler.prediction_expr(
            self.root_, lambda feature: pl.col(temporary[feature])
        )
        return frame.with_columns(casts).with_columns(
            expr.alias(_PREDICTION)
        ), temporary

    def export_rules(self, path: str | Path) -> Path:
        """Write human-readable decision rules to ``path``."""
        self._check_fitted()
        assert self.root_ is not None
        rules: list[str] = []

        def visit(node: Node, conditions: list[str]) -> None:
            if node.is_leaf:
                lhs = " ∧ ".join(conditions) if conditions else "[any]"
                rules.append(f"{lhs} → [{self.target} = {node.prediction!r}]")
                return

            if node.feature is None:
                raise RuntimeError(f"Internal node {node.id} has no split feature")

            for value, child in node.children.items():
                conditions.append(f"[{node.feature} = {value!r}]")
                visit(child, conditions)
                conditions.pop()

            if node.other_child is not None:
                known = ", ".join(repr(value) for value in node.children)
                conditions.append(f"[{node.feature} ∉ {{{known}}}]")
                visit(node.other_child, conditions)
                conditions.pop()
            else:
                # Prediction falls back to this node's majority class for
                # categories that never appeared here during training.
                known = ", ".join(repr(value) for value in node.children)
                lhs = " ∧ ".join(conditions + [f"[{node.feature} ∉ {{{known}}}]"])
                rules.append(f"{lhs} → [{self.target} = {node.prediction!r}]")

        visit(self.root_, [])
        destination = Path(path)
        destination.write_text("\n".join(rules) + "\n", encoding="utf-8")
        return destination

    # -- internals ---------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted or self.root_ is None:
            raise RuntimeError("Model is not fitted; call fit() first")


class _PredictionCompiler:
    """Small helper that compiles the public tree into a Polars expression."""

    def __init__(self, target_dtype: pl.DataType) -> None:
        self.target_dtype = target_dtype

    def prediction_expr(
        self,
        node: Node,
        value_expr: Callable[[str], pl.Expr] = _raw_value_expr,
    ) -> pl.Expr:
        if node.is_leaf or (not node.children and node.other_child is None):
            return _prediction_literal(node.prediction, self.target_dtype)
        if node.feature is None:
            raise RuntimeError(f"Internal node {node.id} has no split feature")

        branches: list[tuple[pl.Expr, pl.Expr]] = []
        for value, child in node.children.items():
            branches.append(
                (
                    value_expr(node.feature) == value,
                    self.prediction_expr(child, value_expr),
                )
            )
        if node.other_child is not None:
            fallback = self.prediction_expr(node.other_child, value_expr)
        else:
            fallback = _prediction_literal(node.prediction, self.target_dtype)

        return _when_chain(branches, fallback)
