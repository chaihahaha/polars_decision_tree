# SPDX-License-Identifier: GPL-3.0-only
"""Polars-based out-of-core ID3 decision trees."""

from .tree import MISSING_SENTINEL, Node, PolarsID3Classifier

__all__ = ["MISSING_SENTINEL", "Node", "PolarsID3Classifier"]
__version__ = "0.1.0"
