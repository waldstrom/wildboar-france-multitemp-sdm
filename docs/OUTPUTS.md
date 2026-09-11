# Output and export reference

[Documentation home](README.md) · [Manuscript crosswalk](MANUSCRIPT_CROSSWALK.md) · [Review guide](../review/README.md)

Output paths below describe existing code conventions. Actual experiment prefixes, engine subdirectories and filenames can vary by runner; the resolved config and output manifest are authoritative.

## Core runs

| Artifact / pattern | Meaning | Publication use |
|---|---|---|
| `exps/<experiment>/input-data-test-<year>/` | Fold-specific stacks and point inputs | Train/test provenance |
| `.../<Season>_stack/selected_predictors.txt` | Retained predictor names/order where exported | Feature provenance; S4/S6b interpretation |
| `output/<Season>/temporal_cv_metrics.csv` | Fold/engine/split performance | Table 3; S6a |
| `output/<Season>/test_<year>/` | Held-out-year models, predictions and diagnostics | Fig. 6; S8 |
| `temporal_cv_metrics_all.csv` | Combined metrics from a meta run | Scenario comparisons |
| `combination_summary_long.csv` / `combination_summary_wide.csv` | Aggregate performance by scenario/engine | Table 3 support |
| `best_runs.csv` | Highest-scoring test fold selected per combination | Diagnostic index; not an all-fold result |
| Saved model objects and predictor matrices | Exact fitted transformation/variable order | Re-evaluation and permutation |
| Novelty summaries and metadata | MESS and NT2 reference statistics | Fig. 8; S9 |

Never combine metrics without retaining scenario, temporal mode, season, engine, test year and split. Training, validation and outer-test rows answer different questions. Fold SD and uncertainty intervals are different summaries; preserve the producing method and number of years.

## Feature importance

The manuscript's five-repeat held-out-year AUC permutation importance is defined by both **what is permuted** and **where it is evaluated**. Changing the repeat count alone does not turn a training-data best-fold diagnostic into the S6b analysis.

| Utility | Available scope |
|---|---|
| [variable_importance.py](../scripts/variable_importance.py) | Permutation, gain and jackknife primitives; caller supplies evaluation data |
| [LOYO meta runner](../scripts/meta_run_leave_one_year.py) | Additional importance on selected best folds |
| [review importance runner](../review/scripts/run_feature_importance.py) | Best fold per engine, evaluated using loaded training/background points |
| [analysis aggregator](../scripts/analyze_feature_importance.py) | Collects supplied ELAPID importance artifacts and summarizes them |
| [review finalizer](../review-finalize-existing-run.py) | All-fold hunting diagnostic for corrected winter monotemporal MaxEnt scenarios |

To reconstruct S6b, inventory all expected years (seven winter, six summer), predictors, evaluated splits and repeats. Preserve raw AUC drops, any normalization to relative percentages, missing-feature treatment and the fold SD calculation. A zero is not interchangeable with a missing or unselected feature. Top-ten main-text tables must derive from the same full rankings.

The [postprocessing config](../configs/postprocessing/analyze_feature_importance.yaml) is an input template. It contains no run locations by default. Relative paths resolve from its own directory. The aggregator does not recover missing fold-level permutation calculations.

## Response curves and maps

[visualization.py](../scripts/visualization.py) supplies plotting helpers used by model runners. Current S7 contains response curves; current S8 contains annual output maps. Inspect grid/mask/year/engine consistency before combining panels. Confirm the actual score transformation, not just a filename ending in `cloglog`.

The 2019 comparison (Fig. 6) is illustrative; the full time series is S8. A difference map (Fig. 7) needs an explicit subtraction direction and common valid-data mask. Upstream ViT classification maps (Fig. 5) are a different product from SDM suitability maps.

## Review finalization

```bash
python review-finalize-existing-run.py --source-run review/output/EXISTING_RUN --n-repeats 5
```

Replace `EXISTING_RUN` with an authorized local completed run. This command does not refit models, but it can recompute saved-model hunting permutations and re-export publication artifacts. It does not supply missing models, stacks or full-feature S6b evaluations.

Saved review folders use stable historical names. [The numbering map](MANUSCRIPT_CROSSWALK.md#historical-export-names-versus-current-numbering) reconciles them with current Table 3, S6a, S6b and S7-S9.

## Storage and provenance

Generated products remain under ignored `exps/`, `outputs/` and `review/output/` locations. Archive permitted outputs with the commit, resolved config, selected features, source checksums, environment and exact command. Use an output manifest to link manuscript items to run files. Research outputs are deliberately absent from Git.
