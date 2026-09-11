# Saved-result analysis configuration

[Config index](../README.md) · [Output reference](../../docs/OUTPUTS.md#feature-importance)

[analyze_feature_importance.yaml](analyze_feature_importance.yaml) is the input template for `python -m scripts.analyze_feature_importance`. Fill in locally available `best_runs` and importance-artifact directories before running it. Empty `meta_runs` is intentional; no research results are distributed.

Paths in this YAML resolve **relative to its own directory**, unlike modelling presets. The default output `../../outputs/feature_importance_analysis` resolves below the repository root's ignored `outputs/`. Absolute paths can be supplied in a local copy.

The helper aggregates existing ELAPID importance artifacts and optional VIF information. It cannot supply missing years or turn training/best-fold calculations into the manuscript's held-out, all-year S6b analysis. Record source files, splits, repeats and missing-feature handling.
