"""Utility helpers for Mahalanobis thinning studies."""

from __future__ import annotations


def _build_trial_params(levels: list[float], variants: list[str]) -> list[dict]:
    """Return ordered Optuna parameter sets.

    The sequence follows the specification:
    1. Iterate over non-global levels (e.g. 0.4, 0.8) for all non-global variants.
    2. Evaluate the global variant for levels ``1.0`` followed by remaining
       levels in descending order.
    """

    levels = sorted(levels)
    non_global = [lvl for lvl in levels if lvl != 1.0]

    params: list[dict] = []
    for lvl in non_global:
        for idx in range(1, len(variants)):
            params.append({"thin_pct": lvl, "thin_variant_idx": idx})

    for lvl in [1.0] + sorted(non_global, reverse=True):
        params.append({"thin_pct": lvl, "thin_variant_idx": 0})

    return params
