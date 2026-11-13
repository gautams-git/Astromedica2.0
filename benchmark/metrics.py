"""Ranking metrics for labelled vectors.

These functions do not read the twelve-patient files. The patients do not
carry adjudicated clinical outcomes, so a metric here is not a clinical score.
"""

from __future__ import annotations


def hit_rate_at_k(rankings: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    return 1.0 if truth.intersection(rankings[:k]) else 0.0


def precision_at_k(rankings: list[str], truth: set[str], k: int) -> float:
    window = rankings[:k]
    if not window:
        return 0.0
    return len([item for item in window if item in truth]) / len(window)


def recall_at_k(rankings: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    return len({item for item in rankings[:k] if item in truth}) / len(truth)


def mean_reciprocal_rank(rankings: list[str], truth: set[str]) -> float:
    for index, label in enumerate(rankings, start=1):
        if label in truth:
            return 1.0 / index
    return 0.0


def rank_correlation(left: list[str], right: list[str]) -> float | None:
    """Spearman correlation on the shared labels. None when fewer than two overlap."""
    shared = [label for label in left if label in right]
    if len(shared) < 2:
        return None
    rank_left = {label: index for index, label in enumerate(left)}
    rank_right = {label: index for index, label in enumerate(right)}
    xs = [rank_left[label] for label in shared]
    ys = [rank_right[label] for label in shared]
    return _spearman(xs, ys)


def _spearman(xs: list[int], ys: list[int]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 1.0 if xs == ys else 0.0
    return num / (den_x * den_y)
