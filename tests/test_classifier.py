# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_decision_tree.tree as tree_module
from polars_decision_tree import PolarsID3Classifier
from polars_decision_tree.tree import Node


def test_fits_xor_perfectly(xor_path: Path) -> None:
    clf = PolarsID3Classifier(target="y", min_samples_leaf=1, min_samples_split=2)
    clf.fit(xor_path, features=["f1", "f2"])

    assert clf.classes_ == ["0", "1"]
    assert clf.root_ is not None
    assert clf.root_.feature == "f1"
    assert not clf.root_.is_leaf
    assert clf.depth_ == 2
    assert clf.n_leaves_ == 4
    assert clf.score(xor_path) == 1.0

    predictions = clf.predict(xor_path).collect()
    assert predictions["prediction"].to_list() == predictions["y"].to_list()


def test_max_depth_zero_is_majority_leaf(tmp_path: Path) -> None:
    path = tmp_path / "d.parquet"
    pl.DataFrame(
        {
            "f": ["a"] * 10 + ["b"] * 6,
            "y": ["0"] * 10 + ["1"] * 6,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", max_depth=0, min_samples_leaf=1)
    clf.fit(path, features=["f"])

    assert clf.root_ is not None and clf.root_.is_leaf
    assert clf.root_.prediction == "0"
    assert clf.depth_ == 0
    assert clf.n_leaves_ == 1
    assert clf.score(path) == 0.625


def test_max_depth_one_stops_before_xor_finishes(xor_path: Path) -> None:
    clf = PolarsID3Classifier(target="y", max_depth=1, min_samples_leaf=1)
    clf.fit(xor_path, features=["f1", "f2"])

    assert clf.depth_ == 1
    assert clf.n_leaves_ == 2
    assert clf.score(xor_path) == 0.9


def test_max_depth_none_fully_grows_xor(xor_path: Path) -> None:
    clf = PolarsID3Classifier(target="y", max_depth=None, min_samples_leaf=1)
    clf.fit(xor_path, features=["f1", "f2"])
    assert clf.depth_ == 2
    assert clf.score(xor_path) == 1.0


def test_classes_come_from_full_scan_not_a_sample(tmp_path: Path) -> None:
    # The rare class appears only after the first 10_000 rows, which is the
    # sample size used by the reference implementation.
    n_common = 10_000
    n_rare = 50
    path = tmp_path / "rare.parquet"
    pl.DataFrame(
        {
            "f": ["common"] * n_common + ["rare"] * n_rare,
            "y": ["a"] * n_common + ["b"] * n_rare,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1)
    clf.fit(path, features=["f"])

    assert clf.classes_ == ["a", "b"]
    assert clf.root_ is not None
    assert clf.root_.feature == "f"
    assert clf.score(path) == 1.0
    assert clf.root_.proba is not None
    np.testing.assert_allclose(clf.root_.proba, [10000 / 10050, 50 / 10050])


def test_missing_feature_values_are_their_own_category(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    pl.DataFrame(
        {
            "f": ["a"] * 6 + ["b"] * 6 + [None] * 6,
            "g": ["x"] * 18,
            "y": ["0"] * 6 + ["1"] * 6 + ["1"] * 6,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1)
    clf.fit(path, features=["f"])

    assert clf.score(path) == 1.0
    assert clf.root_ is not None
    assert not clf.root_.is_leaf
    assert clf.root_.feature == "f"
    # Null is normalised to the documented sentinel.
    from polars_decision_tree import MISSING_SENTINEL

    assert MISSING_SENTINEL in clf.root_.children
    rows = clf.predict(path).collect()
    assert rows.filter(pl.col("f").is_null())["prediction"].to_list() == ["1"] * 6


def test_numeric_features_are_categorical_and_output_dtype_is_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "numeric.parquet"
    pl.DataFrame(
        {
            "f": [1] * 20 + [2] * 20,
            "y": ["no"] * 20 + ["yes"] * 20,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(path, features=["f"])
    assert clf.score(path) == 1.0
    out = clf.predict(path).collect()
    assert out.schema["f"] == pl.Int64
    assert out.schema["prediction"] == pl.String


def test_categorical_target_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "categorical.parquet"
    pl.DataFrame(
        {
            "f": ["a"] * 20 + ["b"] * 20,
            "y": pl.Series(["no"] * 20 + ["yes"] * 20, dtype=pl.Categorical),
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(path, features=["f"])
    assert clf.target_dtype_ == pl.Categorical
    assert clf.classes_ == ["no", "yes"]
    assert clf.score(path) == 1.0
    assert clf.predict(path).collect().schema["prediction"] == pl.Categorical


def test_numeric_target_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "numeric_target.parquet"
    pl.DataFrame(
        {
            "f": ["a"] * 20 + ["b"] * 20,
            "y": pl.Series([0] * 20 + [1] * 20, dtype=pl.UInt16),
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(path, features=["f"])
    assert clf.classes_ == [0, 1]
    assert clf.score(path) == 1.0
    assert clf.predict(path).collect().schema["prediction"] == pl.UInt16


def test_binary_target_has_stable_sorted_classes(tmp_path: Path) -> None:
    path = tmp_path / "bool.parquet"
    pl.DataFrame(
        {
            "f": ["a"] * 20 + ["b"] * 20,
            "y": [False] * 20 + [True] * 20,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(path, features=["f"])
    assert clf.classes_ == [False, True]
    assert clf.score(path) == 1.0


def test_rare_categories_are_grouped_into_other_child(tmp_path: Path) -> None:
    path = tmp_path / "rare_other.parquet"
    pl.DataFrame(
        {
            "f": ["A"] * 100 + ["B"] * 100 + ["C"] * 3 + ["D"] * 3,
            "y": ["0"] * 100 + ["1"] * 106,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=5).fit(path, features=["f"])
    assert clf.root_ is not None
    assert not clf.root_.is_leaf
    assert clf.root_.feature == "f"
    assert set(clf.root_.children) == {"A", "B"}
    assert clf.root_.other_child is not None
    assert clf.root_.other_child.n_samples == 6
    assert clf.root_.other_child.prediction == "1"
    assert clf.score(path) == 1.0


def test_other_branch_below_min_samples_leaf_rejects_split(tmp_path: Path) -> None:
    path = tmp_path / "rare_reject.parquet"
    pl.DataFrame(
        {
            "f": ["A"] * 100 + ["B"] * 100 + ["C"] * 3 + ["D"] * 3,
            "y": ["0"] * 100 + ["1"] * 106,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=10).fit(path, features=["f"])
    assert clf.root_ is not None
    assert clf.root_.is_leaf
    assert clf.root_.prediction == "1"
    assert clf.score(path) == 106 / 206


def test_min_samples_split_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "tiny.parquet"
    pl.DataFrame({"f": ["a", "b"], "y": ["0", "1"]}).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_split=3, min_samples_leaf=1).fit(
        path, features=["f"]
    )
    assert clf.root_ is not None and clf.root_.is_leaf


def test_two_children_must_fit_min_samples_leaf(tmp_path: Path) -> None:
    path = tmp_path / "six.parquet"
    pl.DataFrame(
        {
            "f": ["a", "a", "a", "b", "b", "b"],
            "y": ["0", "0", "0", "1", "1", "1"],
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=4).fit(path, features=["f"])
    assert clf.root_ is not None and clf.root_.is_leaf


def test_gain_and_gain_ratio_can_choose_different_features(tmp_path: Path) -> None:
    # Feature A is pure but has eight branches (split info = 3 bits).
    # Feature B is binary and slightly impure (split info = 1 bit).
    # Their information gains are 1.0 and 0.456..., respectively, so gain
    # picks A while gain ratio picks B.
    y = ["0"] * 8 + ["1"] * 8
    a = ["a1", "a1", "a2", "a2", "a3", "a3", "a4", "a4"] + [
        "a5",
        "a5",
        "a6",
        "a6",
        "a7",
        "a7",
        "a8",
        "a8",
    ]
    b = ["x", "x", "x", "x", "x", "x", "x", "y"] + [
        "x",
        "y",
        "y",
        "y",
        "y",
        "y",
        "y",
        "y",
    ]
    path = tmp_path / "criteria.parquet"
    pl.DataFrame({"A": a, "B": b, "y": y}).write_parquet(path)

    gain_clf = PolarsID3Classifier(
        target="y", criterion="gain", min_samples_leaf=1
    ).fit(path, features=["A", "B"])
    ratio_clf = PolarsID3Classifier(
        target="y", criterion="gain_ratio", min_samples_leaf=1
    ).fit(path, features=["A", "B"])

    assert gain_clf.root_ is not None and gain_clf.root_.feature == "A"
    assert ratio_clf.root_ is not None and ratio_clf.root_.feature == "B"
    # Both criteria split the same root differently; sanity-check their gains.
    assert gain_clf.root_.gain == pytest.approx(1.0)
    assert ratio_clf.root_.gain == pytest.approx(
        1.0 - (-(1 / 8) * np.log2(1 / 8) - (7 / 8) * np.log2(7 / 8))
    )


def test_feature_order_breaks_gain_ties(tmp_path: Path) -> None:
    path = tmp_path / "tie.parquet"
    pl.DataFrame(
        {
            "first": ["a"] * 20 + ["b"] * 20,
            "second": ["c"] * 20 + ["d"] * 20,
            "y": ["0"] * 20 + ["1"] * 20,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(
        path, features=["second", "first"]
    )
    assert clf.root_ is not None and clf.root_.feature == "second"


def test_feature_inference_uses_every_other_column(tmp_path: Path) -> None:
    path = tmp_path / "infer.parquet"
    pl.DataFrame(
        {
            "f1": ["a"] * 20 + ["b"] * 20,
            "f2": ["constant"] * 40,
            "y": ["0"] * 20 + ["1"] * 20,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(path)
    assert clf.feature_names_ == ["f1", "f2"]
    assert clf.root_ is not None and clf.root_.feature == "f1"


def test_directory_and_file_list_sources(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "f": ["a"] * 20 + ["b"] * 20,
            "y": ["0"] * 20 + ["1"] * 20,
        }
    )
    df.head(20).write_parquet(tmp_path / "part1.parquet")
    df.tail(20).write_parquet(tmp_path / "part2.parquet")

    clf_dir = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(
        tmp_path, features=["f"]
    )
    assert clf_dir.score(tmp_path) == 1.0

    clf_files = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(
        [tmp_path / "part1.parquet", tmp_path / "part2.parquet"], features=["f"]
    )
    assert (
        clf_files.score([tmp_path / "part1.parquet", tmp_path / "part2.parquet"]) == 1.0
    )


def test_one_level_scan_per_depth(
    xor_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[int]] = []
    original = tree_module._LevelBuilder._scan_level

    def wrapped(self, frontier):  # type: ignore[no-untyped-def]
        calls.append([node.id for node in frontier])
        return original(self, frontier)

    monkeypatch.setattr(tree_module._LevelBuilder, "_scan_level", wrapped)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(
        xor_path, features=["f1", "f2"]
    )

    assert clf.depth_ == 2
    assert len(calls) == 2  # levels 0 and 1; level 2 is pre-pruned as pure
    assert calls[0] == [0]
    assert sorted(calls[1]) == sorted([1, 2])
    assert clf.scan_counts_ == {
        "root": 1,
        "level": 2,
        "validation": 0,
        "total": 3,
    }


def test_predict_is_lazy_and_can_sink(tmp_path: Path, xor_path: Path) -> None:
    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(
        xor_path, features=["f1", "f2"]
    )
    lazy = clf.predict(xor_path)
    assert isinstance(lazy, pl.LazyFrame)

    output = tmp_path / "predictions.parquet"
    lazy.sink_parquet(output)
    result = pl.read_parquet(output)
    assert result["prediction"].to_list() == result["y"].to_list()


def test_export_rules(tmp_path: Path, xor_path: Path) -> None:
    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(
        xor_path, features=["f1", "f2"]
    )
    rules_path = clf.export_rules(tmp_path / "rules.txt")
    assert rules_path.exists()
    text = rules_path.read_text(encoding="utf-8")
    assert "→ [y =" in text
    # The file contains one rule per leaf plus explicit fallback rules for
    # categories that were unseen during training.
    assert text.count("→") >= clf.n_leaves_
    assert "f1" in text and "f2" in text
    assert isinstance(clf.root_, Node)


def test_empty_score_raises(tmp_path: Path, xor_path: Path) -> None:
    empty = tmp_path / "empty.parquet"
    pl.DataFrame(
        {"f1": [], "f2": [], "y": []},
        schema={"f1": pl.String, "f2": pl.String, "y": pl.String},
    ).write_parquet(empty)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(
        xor_path, features=["f1", "f2"]
    )
    with pytest.raises(ValueError, match="empty"):
        clf.score(empty)


def test_unfitted_estimator_raises(xor_path: Path) -> None:
    clf = PolarsID3Classifier(target="y")
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(xor_path)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.score(xor_path)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.export_rules("rules.txt")


def test_invalid_hyperparameters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="criterion"):
        PolarsID3Classifier(target="y", criterion="gini").fit(
            tmp_path / "does_not_matter.parquet"
        )
    with pytest.raises(ValueError, match="max_depth"):
        PolarsID3Classifier(target="y", max_depth=-1).fit(
            tmp_path / "does_not_matter.parquet"
        )
    with pytest.raises(ValueError, match="min_samples_leaf"):
        PolarsID3Classifier(target="y", min_samples_leaf=0).fit(
            tmp_path / "does_not_matter.parquet"
        )


def test_validation_of_columns(tmp_path: Path) -> None:
    path = tmp_path / "cols.parquet"
    pl.DataFrame({"f": ["a", "b"], "y": ["0", "1"]}).write_parquet(path)

    with pytest.raises(ValueError, match="not in the parquet schema"):
        PolarsID3Classifier(target="missing").fit(path, features=["f"])
    with pytest.raises(ValueError, match="At least one feature"):
        PolarsID3Classifier(target="y").fit(path, features=[])
    with pytest.raises(ValueError, match="cannot also be a feature"):
        PolarsID3Classifier(target="y").fit(path, features=["y"])
    with pytest.raises(ValueError, match="not found"):
        PolarsID3Classifier(target="y").fit(path, features=["nope"])
    with pytest.raises(ValueError, match="unique"):
        PolarsID3Classifier(target="y").fit(path, features=["f", "f"])
    reserved_path = tmp_path / "reserved.parquet"
    pl.DataFrame({"f": ["a", "b"], "prediction": ["0", "1"]}).write_parquet(
        reserved_path
    )
    with pytest.raises(ValueError, match="reserved"):
        PolarsID3Classifier(target="prediction").fit(reserved_path, features=["f"])


def test_null_target_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "null_target.parquet"
    pl.DataFrame({"f": ["a", "b"], "y": ["0", None]}).write_parquet(path)
    with pytest.raises(ValueError, match="null"):
        PolarsID3Classifier(target="y").fit(path, features=["f"])


def test_nested_target_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nested.parquet"
    pl.DataFrame(
        {
            "f": ["a", "b"],
            "y": [[1], [2]],
        },
        schema={"f": pl.String, "y": pl.List(pl.Int64)},
    ).write_parquet(path)
    with pytest.raises(ValueError, match="scalar"):
        PolarsID3Classifier(target="y").fit(path, features=["f"])


def test_constant_feature_and_single_class(tmp_path: Path) -> None:
    path = tmp_path / "constant.parquet"
    pl.DataFrame({"f": ["a"] * 20, "y": ["only"] * 20}).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(path, features=["f"])
    assert clf.root_ is not None and clf.root_.is_leaf
    assert clf.classes_ == ["only"]
    assert clf.depth_ == 0
    assert clf.score(path) == 1.0


def test_min_gain_can_stop_a_perfect_split(tmp_path: Path) -> None:
    path = tmp_path / "min_gain.parquet"
    pl.DataFrame(
        {
            "f": ["a"] * 20 + ["b"] * 20,
            "y": ["0"] * 20 + ["1"] * 20,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1, min_gain=1.0).fit(
        path, features=["f"]
    )
    assert clf.root_ is not None and clf.root_.is_leaf


def test_multiclass_tree(tmp_path: Path) -> None:
    path = tmp_path / "multiclass.parquet"
    pl.DataFrame(
        {
            "f": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
            "y": ["red"] * 20 + ["green"] * 20 + ["blue"] * 20,
        }
    ).write_parquet(path)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(path, features=["f"])
    assert clf.classes_ == ["blue", "green", "red"]
    assert clf.score(path) == 1.0
    assert clf.root_ is not None and len(clf.root_.children) == 3


def test_nested_feature_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nested_feature.parquet"
    pl.DataFrame(
        {"f": [[1], [2]], "y": ["0", "1"]},
        schema={"f": pl.List(pl.Int64), "y": pl.String},
    ).write_parquet(path)

    with pytest.raises(ValueError, match="scalar"):
        PolarsID3Classifier(target="y").fit(path, features=["f"])


def test_predict_rejects_missing_features(tmp_path: Path, xor_path: Path) -> None:
    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(
        xor_path, features=["f1", "f2"]
    )
    path = tmp_path / "only_f1.parquet"
    pl.DataFrame({"f1": ["a"]}).write_parquet(path)
    with pytest.raises(ValueError, match="missing features"):
        clf.predict(path)


def test_predict_does_not_require_target_and_preserves_extra_columns(
    tmp_path: Path, xor_path: Path
) -> None:
    clf = PolarsID3Classifier(target="y", min_samples_leaf=1).fit(
        xor_path, features=["f1", "f2"]
    )
    path = tmp_path / "no_target.parquet"
    pl.DataFrame(
        {
            "f1": ["a", "b"],
            "f2": ["x", "x"],
            "extra": [1, 2],
        }
    ).write_parquet(path)

    result = clf.predict(path).collect()
    assert result.columns == ["f1", "f2", "extra", "prediction"]
    assert result["prediction"].to_list() == ["0", "1"]

    with pytest.raises(ValueError, match="required columns"):
        clf.score(path)


def test_unseen_category_uses_other_child(tmp_path: Path) -> None:
    train = tmp_path / "other_train.parquet"
    pl.DataFrame(
        {
            "f": ["A"] * 100 + ["B"] * 100 + ["C"] * 3 + ["D"] * 3,
            "y": ["0"] * 100 + ["1"] * 106,
        }
    ).write_parquet(train)

    clf = PolarsID3Classifier(target="y", min_samples_leaf=5).fit(train, features=["f"])
    unseen = tmp_path / "unseen.parquet"
    pl.DataFrame({"f": ["Z"]}).write_parquet(unseen)
    assert clf.predict(unseen).collect()["prediction"].to_list() == ["1"]


def test_features_must_be_a_sequence(tmp_path: Path) -> None:
    path = tmp_path / "cols2.parquet"
    pl.DataFrame({"f": ["a", "b"], "y": ["0", "1"]}).write_parquet(path)
    with pytest.raises(TypeError, match="sequence"):
        PolarsID3Classifier(target="y").fit(path, features="f")  # type: ignore[arg-type]


def test_validation_missing_columns_is_rejected_before_building(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cols3.parquet"
    pl.DataFrame({"f": ["a", "b"], "y": ["0", "1"]}).write_parquet(path)
    validation = tmp_path / "bad_validation.parquet"
    pl.DataFrame({"f": ["a"]}).write_parquet(validation)

    with pytest.raises(ValueError, match="Validation data is missing"):
        PolarsID3Classifier(target="y", prune=True).fit(
            path, features=["f"], eval_path=validation
        )


def test_holdout_accuracy_generalizes(tmp_path: Path) -> None:
    # Independent train and test samples, both from the same noisy concept.
    # This checks generalization rather than merely re-scoring the training
    # rows used to build the tree.
    rng = np.random.default_rng(7)

    def make_rows(n: int) -> pl.DataFrame:
        f0 = rng.integers(0, 4, size=n).astype(str)
        f1 = rng.integers(0, 4, size=n).astype(str)
        f2 = rng.integers(0, 4, size=n).astype(str)
        signal = (
            (f0 == "0").astype(np.int8)
            + (f1 == "1").astype(np.int8)
            + (f2 == "2").astype(np.int8)
        )
        labels = np.where(signal >= 2, "yes", "no")
        flip = rng.random(n) < 0.08
        labels[flip] = np.where(labels[flip] == "yes", "no", "yes")
        return pl.DataFrame({"f0": f0, "f1": f1, "f2": f2, "y": labels})

    train = make_rows(60_000)
    test = make_rows(15_000)
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train.write_parquet(train_path)
    test.write_parquet(test_path)

    clf = PolarsID3Classifier(
        target="y",
        criterion="gain_ratio",
        max_depth=5,
        min_samples_split=100,
        min_samples_leaf=50,
    ).fit(train_path, features=["f0", "f1", "f2"])

    train_accuracy = clf.score(train_path)
    test_accuracy = clf.score(test_path)
    majority = max(test["y"].value_counts()["count"]) / test.height

    assert test_accuracy > 0.85
    assert test_accuracy < 0.97
    assert train_accuracy - test_accuracy < 0.06
    assert test_accuracy > majority + 0.05
