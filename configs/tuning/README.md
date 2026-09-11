# Tuning and sensitivity experiments

[Config index](../README.md) · [Tuning implementation audit](../../docs/REPRODUCIBILITY.md#tuning-and-outer-evaluation)

These are separate historical searches, not an automatic inner stage of the LOYO meta runners. Inspect the objective's evaluation data before reuse: `multitemporal_optuna.py` and `multitemporal_optuna_regularization.py` use test-year metrics in trial selection as implemented. They must not be represented as independent outer-test evaluation.

| Configuration | Consumer |
|---|---|
| [configtuning.yaml](configtuning.yaml) | `scripts.optuna_tuning`; also `scripts.optuna_gbm` with compatible settings |
| [configmultitemporal_optuna.yaml](configmultitemporal_optuna.yaml) | `scripts.multitemporal_optuna` |
| [configmultitemporal_optuna_regularization_winter.yaml](configmultitemporal_optuna_regularization_winter.yaml) | `scripts.multitemporal_optuna_regularization` |
| [configmultitemporal_optuna_regularization_summer.yaml](configmultitemporal_optuna_regularization_summer.yaml) | `scripts.multitemporal_optuna_regularization` |
| [configmaha_thin_optuna_winter.yaml](configmaha_thin_optuna_winter.yaml) | `scripts/maha-thin-multi.py` |
| [configmaha_thin_optuna_summer.yaml](configmaha_thin_optuna_summer.yaml) | `scripts/maha-thin-multi.py` |

The two summer configs' `presence_data.thinning.percentages` blocks had invalid YAML indentation. That syntax is repaired; their scalar settings are preserved.

S5 includes example MaxEnt/GBM search ranges. The checked-in search presets belong to different experiments and are not a verified transcript of every final fold's search. Match the configuration to its consumer and saved trial records rather than assuming all search keys are universally supported. In particular, the GBM calibration implementation supports Platt, not every choice printed in S5.

Archive the study database, trials, chosen objective, split membership and resolved best parameters. Keep final outer test years out of parameter selection when designing a new nested evaluation.
