# Configuration path migration

[Configuration guide](CONFIGURATION.md) · [Config index](../configs/README.md)

The September 2026 tidy-up moves configuration files out of the repository root and groups them by purpose. Numerical settings in migrated files are preserved. Two summer tuning files had invalid indentation in `presence_data.thinning.percentages`; that indentation is repaired. The postprocessing output directory now resolves into the root `outputs/` folder.

Update external commands and local scripts to these paths. The two LOYO meta runners now default to the reviewed presets; pass a historical config explicitly to retain the previous selection.

| Previous path | Current path |
|---|---|
| `config.yaml` | [configs/legacy/config.yaml](../configs/legacy/config.yaml) |
| `configall.yaml` | [configs/legacy/configall.yaml](../configs/legacy/configall.yaml) |
| `configsummer.yaml` | [configs/legacy/configsummer.yaml](../configs/legacy/configsummer.yaml) |
| `configwinter.yaml` | [configs/legacy/configwinter.yaml](../configs/legacy/configwinter.yaml) |
| `config_gbm-compare.yaml` | [configs/experiments/config_gbm-compare.yaml](../configs/experiments/config_gbm-compare.yaml) |
| `config_gbm_loyo_compare.yaml` | [configs/experiments/config_gbm_loyo_compare.yaml](../configs/experiments/config_gbm_loyo_compare.yaml) |
| `config_gbm_loyo_gbif.yaml` | [configs/experiments/config_gbm_loyo_gbif.yaml](../configs/experiments/config_gbm_loyo_gbif.yaml) |
| `config_gbm_loyo_gbif_grid.yaml` | [configs/experiments/config_gbm_loyo_gbif_grid.yaml](../configs/experiments/config_gbm_loyo_gbif_grid.yaml) |
| `configmultitemporal_winter.yaml` | [configs/experiments/configmultitemporal_winter.yaml](../configs/experiments/configmultitemporal_winter.yaml) |
| `configmultitemporal_summer.yaml` | [configs/experiments/configmultitemporal_summer.yaml](../configs/experiments/configmultitemporal_summer.yaml) |
| `configwinter_monotemporal.yaml` | [configs/experiments/configwinter_monotemporal.yaml](../configs/experiments/configwinter_monotemporal.yaml) |
| `configsummer_monotemporal.yaml` | [configs/experiments/configsummer_monotemporal.yaml](../configs/experiments/configsummer_monotemporal.yaml) |
| `configtuning.yaml` | [configs/tuning/configtuning.yaml](../configs/tuning/configtuning.yaml) |
| `configmaha_thin_optuna_summer.yaml` | [configs/tuning/configmaha_thin_optuna_summer.yaml](../configs/tuning/configmaha_thin_optuna_summer.yaml) |
| `configmaha_thin_optuna_winter.yaml` | [configs/tuning/configmaha_thin_optuna_winter.yaml](../configs/tuning/configmaha_thin_optuna_winter.yaml) |
| `configmultitemporal_optuna.yaml` | [configs/tuning/configmultitemporal_optuna.yaml](../configs/tuning/configmultitemporal_optuna.yaml) |
| `configmultitemporal_optuna_regularization_summer.yaml` | [configs/tuning/configmultitemporal_optuna_regularization_summer.yaml](../configs/tuning/configmultitemporal_optuna_regularization_summer.yaml) |
| `configmultitemporal_optuna_regularization_winter.yaml` | [configs/tuning/configmultitemporal_optuna_regularization_winter.yaml](../configs/tuning/configmultitemporal_optuna_regularization_winter.yaml) |
| `featureconfig.yaml` | [configs/features/featureconfig.yaml](../configs/features/featureconfig.yaml) |
| `scripts/analyze_feature_importance_config.yaml` | [configs/postprocessing/analyze_feature_importance.yaml](../configs/postprocessing/analyze_feature_importance.yaml) |

## Legacy scenario matrix

All 44 files formerly at `configs/<year>-<season>_<source>.yaml` are now under [configs/legacy/scenarios](../configs/legacy/scenarios). Basenames and parameter values are preserved.

| Previous example | Current path |
|---|---|
| `configs/2022-Winter_ALL.yaml` | [configs/legacy/scenarios/2022-Winter_ALL.yaml](../configs/legacy/scenarios/2022-Winter_ALL.yaml) |

The matrix covers Winter 2017-2022 and Summer 2018-2022 with ALL, GBIF, RRN and SNCF variants. It is not the final LOYO year matrix (which extends to 2023). Sequential and parallel season testers now read only this subdirectory.

## New configurations

| File | Relationship to historical sources |
|---|---|
| [loyo_all_sources.yaml](../configs/reviewed/loyo_all_sources.yaml) | All-source aggregate with corrected winter hunting, five repeats and distinct run names |
| [loyo_gbif.yaml](../configs/reviewed/loyo_gbif.yaml) | GBIF aggregate with the same explicit changes |

The machine-readable [catalog](../configs/catalog.yaml) records every individual legacy path and current runner. No shared-loader inheritance or parameter harmonization was introduced.
