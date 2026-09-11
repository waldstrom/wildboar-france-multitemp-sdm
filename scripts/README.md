# Script index

[Repository home](../README.md) · [Config index](../configs/README.md) · [Methods](../docs/METHODS.md)

Use `python -m scripts.<module>` from the repository root where the filename is a normal Python module. Pass the correct configuration schema; aggregate configs are not accepted by the direct seasonal model runners.

## Entry points

| Script | Purpose | Configuration / guide |
|---|---|---|
| [meta_run_leave_one_year.py](meta_run_leave_one_year.py) | All-source LOYO comparison; both engines | [Reviewed all-source preset](../configs/reviewed/loyo_all_sources.yaml) |
| [meta_run_leave_one_year_gbif.py](meta_run_leave_one_year_gbif.py) | GBIF-only LOYO comparison | [Reviewed GBIF preset](../configs/reviewed/loyo_gbif.yaml) |
| [multitemporal.py](multitemporal.py) | Seasonal model retaining annual predictors | Extract an inner scenario; [quick start](../docs/QUICKSTART.md#run-one-seasonal-scenario-first) |
| [monotemporal.py](monotemporal.py) | Seasonal training-year mean baseline | Extract an inner monotemporal scenario |
| [meta_run.py](meta_run.py) | Historical fixed-year engine comparison | [Experiment configs](../configs/experiments/README.md) |
| [run_pipeline.py](run_pipeline.py) | Original step-controlled pipeline | [Legacy configs](../configs/legacy/README.md) |
| [seasontester.py](seasontester.py), [seasontester_parallel.py](seasontester_parallel.py) | Execute the 44 legacy scenario snapshots | `configs/legacy/scenarios/`; `--dry` creates local configs |
| [optuna_tuning.py](optuna_tuning.py), [optuna_gbm.py](optuna_gbm.py) | Separate engine searches | [Tuning guide](../configs/tuning/README.md) |
| [multitemporal_optuna.py](multitemporal_optuna.py), [multitemporal_optuna_regularization.py](multitemporal_optuna_regularization.py) | Historical multitemporal search experiments | [Objective-data caveat](../docs/REPRODUCIBILITY.md#tuning-and-outer-evaluation) |
| [maha-thin-multi.py](maha-thin-multi.py), [bias-strength-multi-optuna.py](bias-strength-multi-optuna.py), [meta_run_grid_optuna.py](meta_run_grid_optuna.py) | Thinning, bias and grid sensitivity searches | Exploratory; inspect script/config before execution |
| [analyze_feature_importance.py](analyze_feature_importance.py) | Aggregate saved ELAPID importance artifacts | [Postprocessing config](../configs/postprocessing/README.md) |
| [mess_evaluation.py](mess_evaluation.py) | Novelty diagnostics and run comparison | `--help`; [threshold reference](../docs/REPRODUCIBILITY.md#novelty-threshold) |
| [validate_repository.py](validate_repository.py) | Reject tracked research data and artifacts | Python standard library |
| [validate_documentation.py](validate_documentation.py) | Check YAML/catalog, links and core metadata | Python + PyYAML |

## Processing modules

| Module | Responsibility | Study reference |
|---|---|---|
| [preprocessing.py](preprocessing.py), [multitemp_preprocessing.py](multitemp_preprocessing.py) | Raster preparation/export | Fig. 4; S4 |
| [thinning.py](thinning.py), [maha_thin_utils.py](maha_thin_utils.py) | Environmental/grid thinning | S1 |
| [bias_grid.py](bias_grid.py), [background_sampling.py](background_sampling.py) | Prepared bias inputs and background sampling | Fig. 3; S2 |
| [variable_selection.py](variable_selection.py), [correlation_analysis.py](correlation_analysis.py) | Predictor redundancy and selection | S4 |
| [maxent_training.py](maxent_training.py), [gbm_training.py](gbm_training.py) | Engine fitting and predictions | S5 |
| [cross_validation.py](cross_validation.py), [evaluation.py](evaluation.py) | Additional CV and metrics | Table 3; S6a |
| [variable_importance.py](variable_importance.py) | Importance primitives | Tables 4-5; S6b |
| [visualization.py](visualization.py) | Maps, response curves and diagnostics | Fig. 6; S7-S8 |
| [utils.py](utils.py) | Configuration, raster, point and export helpers | Shared infrastructure |

## Specialized and historical utilities

`multitemporal_classifier_control.py`, `positive_control_zero_others.py` and `positive_control_utils.py` implement SDM positive controls. The classifier-control filename does not denote the Vision Transformer training pipeline. `seasoncombinedtest.py`, `correct_experiments.py`, `science.py`, `fix_geodata.py` and `missing_data_interpolator.py` are specialized experiment/preparation helpers; inspect their entry points and local-path expectations before use.

The root [review-corrections.py](../review-corrections.py) and [review-finalize-existing-run.py](../review-finalize-existing-run.py) have their own [review guide](../review/README.md). Export helpers are covered in [Public releases](../docs/PUBLIC_RELEASE.md).
