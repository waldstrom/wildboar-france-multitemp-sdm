# Reproduction status and implementation differences

[Documentation home](README.md) · [Crosswalk](MANUSCRIPT_CROSSWALK.md) · [Configuration](CONFIGURATION.md)

This public repository supports inspecting methods and running them with prepared local inputs. Exact numerical reproduction additionally needs the authors' resolved run configurations, software environment, input releases, selected predictors, fitted models and output manifests. The reviewed PDFs and source code alone do not establish that a particular historical run used every setting now described in Methods.

## What this update establishes

| Available here | Requires separate run/input records |
|---|---|
| Source code, tests and explicit config families | Prepared source data and their access permissions |
| Corrected winter candidate settings in reviewed examples | Corrected fitted models and fold-specific selected features |
| YAML and documentation validation | Identical dependency/native-library versions |
| Methods-to-code and figure-to-output cross-references | Final numerical tables and map products |
| Software licence and citation metadata | An archived software release DOI, if later minted |

## Tuning and outer evaluation

The manuscript (lines 347-358) describes spatially blocked Optuna tuning inside training data, followed by outer temporal LOYO testing. The public implementation contains several different workflows:

| Code path | Observed behavior | Reproduction consequence |
|---|---|---|
| [meta_run_leave_one_year.py](../scripts/meta_run_leave_one_year.py) | Invokes seasonal models using supplied parameters; can perform additional best-run diagnostics | Does not itself launch nested Optuna within each LOYO fold |
| [optuna_tuning.py](../scripts/optuna_tuning.py), `objective` | Uses `GeographicKFold(n_splits=5)` on its prepared presence data | Inner geographical tuning component; verify exclusion of the outer year in actual input/run records |
| [multitemporal_optuna.py](../scripts/multitemporal_optuna.py), `_objective` | Returns the sum of AUC and CBI from `test_metrics` | Exploratory test-year tuning; not independent outer-test evaluation |
| [multitemporal_optuna_regularization.py](../scripts/multitemporal_optuna_regularization.py), `_objective` | Selects its objective from `test_metrics` | Same distinction: not a leakage-safe nested LOYO recipe as written |
| Reviewed aggregate YAMLs | Contain fixed inherited engine parameters; additional CV disabled | Preserve fold-specific best-parameter records if reproducing a tuned analysis |

The historical search scripts remain under their original algorithms so the public snapshot is traceable. This documentation update does not retrofit a new tuning design or claim that historic results are invalid. Establish which code and resolved parameters produced the paper from the original run records. Do not report outer-test scores as independent if those same scores selected parameters.

S5 lists an `isotonic` calibration choice, but the inspected [GBM calibration helper](../scripts/gbm_training.py) supports Platt calibration only. The published search table therefore does not imply every listed option is executable in this snapshot. Preserve actual trial records and supported-method choices.

## Feature-importance coverage

The manuscript specifies five repeats on held-out LOYO data, followed by averaging across all available test years (Methods lines 369-373; S6b). The reviewed presets set five repeats, but the consuming utility determines the data split and fold scope:

- [meta_run_leave_one_year.py](../scripts/meta_run_leave_one_year.py) selects best folds for additional importance diagnostics, with a separate repeat-count argument.
- [review/scripts/run_feature_importance.py](../review/scripts/run_feature_importance.py) selects the best test-scoring fold per engine, then loads training presence/background points for its importance calculation.
- [review-finalize-existing-run.py](../review-finalize-existing-run.py) supplies an all-fold, MaxEnt-only **hunting** diagnostic for corrected winter monotemporal scenarios. It is not a complete all-feature exporter for all study scenarios.
- [analyze_feature_importance.py](../scripts/analyze_feature_importance.py) aggregates supplied importance artifacts; it cannot create missing held-out-year evaluations merely by averaging available files.

Before rebuilding S6b, require per-feature/per-year records with scenario, engine, evaluated split, repeats and selected-feature provenance. A best-fold diagnostic, or an all-fold hunting-only table, is not the complete S6b analysis. Missing/not-selected values and zero importance must be distinguished explicitly.

## Novelty threshold

The manuscript describes ExDet-NT2 scaled by the **maximum training distance**, with NT2 > 1 flagged (lines 375-383; Fig. 8). The inspected [mess_evaluation.py](../scripts/mess_evaluation.py) computes squared Mahalanobis distances and scales them by a configurable training quantile, default `NT2_QUANTILE = 0.99`. The [review wrapper](../review/scripts/run_novelty.py) uses that default.

A 99th-percentile denominator and a maximum denominator are different reference thresholds. `--nt2-quantile 1.0` selects the maximum of the available finite training squared distances for a direct novelty invocation. This identifies a parameter setting, not evidence that the publication used it. Check saved `nt2_quantile` and `nt2_threshold_md2` metadata before associating exports with Fig. 8 or S9. Do not change a threshold simply to match a desired result.

## Engine scales

MaxEnt uses its configured cloglog output. LightGBM's wrapper predicts its native presence-background score or a Platt-adjusted score. Some shared raster filenames contain `cloglog` regardless of the engine. A filename is not evidence of the mathematical transformation. [Engine interpretation](gbm_maxent_alignment.md) explains this distinction.

## Upstream data and Vision Transformer

The manuscript's land-cover stage uses a fine-tuned ViT, reference labels, training/inference settings and classification evaluation (Fig. 5, Table 2, S3). This public SDM snapshot consumes derived land-cover layers; it does not include the full classifier training workflow, labels, image chips or trained weights. `multitemporal_classifier_control.py` is an SDM positive-control experiment, not the ViT trainer.

Similarly, source-quality filtering and some traffic/accessibility bias-map construction require upstream records. Consult [data acquisition](../data/README.md), not a blanket assumption that all inputs can be regenerated by the SDM runner.

## Archiving a reproducible run

Preserve, subject to each source's access terms:

1. Git commit, dirty-state patch if applicable, Python and native-library environment.
2. Resolved scenario YAML, source config, feature rules and exact per-fold tuned parameters.
3. Input identifiers/releases, checksums, grid/units, season assignment and preprocessing commands.
4. Train/validation/test membership, random seeds, occurrence/background counts and bias settings.
5. Selected predictor names/order, fitted model metadata and stack provenance for every fold.
6. Metrics, importance repeats and evaluated split, response-curve settings, novelty thresholds and masks.
7. Output manifest linking each table/figure to the producing run and command.

A successful repository check certifies code/config/documentation consistency at its stated scope. It does not replace this scientific provenance.
