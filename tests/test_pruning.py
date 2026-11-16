# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from pathlib import Path

import polars as pl

from polars_decision_tree import PolarsID3Classifier


def _training_split(tmp_path: Path) -> Path:
    path = tmp_path / "train.parquet"
    pl.DataFrame(
        {
            "f": ["a"] * 60 + ["b"] * 40,
            "y": ["no"] * 60 + ["yes"] * 40,
        }
    ).write_parquet(path)
    return path


def test_reduced_error_pruning_removes_useless_subtree(tmp_path: Path) -> None:
    train = _training_split(tmp_path)
    validation = tmp_path / "validation.parquet"
    pl.DataFrame({"f": ["a", "b"] * 20, "y": ["no"] * 40}).write_parquet(validation)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1, prune=True)
    clf.fit(train, features=["f"], eval_path=validation)

    assert clf.root_ is not None
    assert clf.root_.is_leaf
    assert clf.root_.prediction == "no"
    assert clf.depth_ == 0
    assert clf.n_leaves_ == 1
    assert clf.score(validation) == 1.0
    # The tree was genuinely useful on the training set before pruning.
    assert clf.score(train) == 0.6


def test_reduced_error_pruning_keeps_useful_subtree(tmp_path: Path) -> None:
    train = _training_split(tmp_path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1, prune=True)
    clf.fit(train, features=["f"], eval_path=train)

    assert clf.root_ is not None
    assert not clf.root_.is_leaf
    assert clf.root_.feature == "f"
    assert clf.depth_ == 1
    assert clf.n_leaves_ == 2


def test_no_eval_path_means_no_pruning(tmp_path: Path) -> None:
    train = _training_split(tmp_path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1, prune=True)
    clf.fit(train, features=["f"])

    assert clf.root_ is not None
    assert not clf.root_.is_leaf
    assert clf.depth_ == 1


def test_prune_false_keeps_tree_even_with_eval(tmp_path: Path) -> None:
    train = _training_split(tmp_path)
    validation = tmp_path / "validation.parquet"
    pl.DataFrame({"f": ["a", "b"] * 20, "y": ["no"] * 40}).write_parquet(validation)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1, prune=False)
    clf.fit(train, features=["f"], eval_path=validation)

    assert clf.root_ is not None
    assert not clf.root_.is_leaf
    assert clf.depth_ == 1


def test_pruning_handles_unseen_validation_categories(tmp_path: Path) -> None:
    train = _training_split(tmp_path)
    validation = tmp_path / "unseen.parquet"
    # Every validation row has a category never seen at the root split.  It
    # terminates at the root and should not crash or be counted as reaching a
    # child.
    pl.DataFrame({"f": ["c", "d"] * 25, "y": ["no"] * 50}).write_parquet(validation)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1, prune=True)
    clf.fit(train, features=["f"], eval_path=validation)

    assert clf.root_ is not None
    assert clf.root_.is_leaf
    assert clf.score(validation) == 1.0


def test_pruning_with_empty_validation_keeps_tree(tmp_path: Path) -> None:
    train = _training_split(tmp_path)
    validation = tmp_path / "empty.parquet"
    pl.DataFrame(
        {"f": [], "y": []}, schema={"f": pl.String, "y": pl.String}
    ).write_parquet(validation)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1, prune=True)
    clf.fit(train, features=["f"], eval_path=validation)

    assert clf.root_ is not None
    assert not clf.root_.is_leaf
    assert clf.depth_ == 1


def test_pruning_is_applied_bottom_up(tmp_path: Path) -> None:
    # Training needs two splits to fit the XOR interaction.  Validation labels
    # depend only on f1, so the root split is useful while both second-level
    # splits are not.  A correct bottom-up pass keeps the root and prunes both
    # children, leaving a depth-1 tree.
    train = tmp_path / "train.parquet"
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
    pl.DataFrame({"f1": f1, "f2": f2, "y": y}).write_parquet(train)

    validation = tmp_path / "validation.parquet"
    pl.DataFrame(
        {
            "f1": ["a"] * 40 + ["b"] * 40,
            "f2": ["x", "y"] * 40,
            "y": ["1"] * 40 + ["0"] * 40,
        }
    ).write_parquet(validation)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1, prune=True)
    clf.fit(train, features=["f1", "f2"], eval_path=validation)

    assert clf.root_ is not None
    assert not clf.root_.is_leaf
    assert clf.root_.feature == "f1"
    assert clf.depth_ == 1
    assert clf.n_leaves_ == 2
    assert clf.score(train) == 0.9
    assert clf.score(validation) == 1.0
    assert clf.scan_counts_ == {
        "root": 1,
        "level": 2,
        "validation": 1,
        "total": 4,
    }
