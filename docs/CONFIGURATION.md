# Configuration reference

[Documentation home](README.md) · [Config index](../configs/README.md) · [Migration table](CONFIG_MIGRATION.md)

## Choose by task

| Task | Configuration | Consumer |
|---|---|---|
| Reviewed all-source LOYO starting point | [loyo_all_sources.yaml](../configs/reviewed/loyo_all_sources.yaml) | `scripts.meta_run_leave_one_year` |
| Reviewed GBIF-only LOYO starting point | [loyo_gbif.yaml](../configs/reviewed/loyo_gbif.yaml) | `scripts.meta_run_leave_one_year_gbif` |
| Historical seasonal/engine comparisons | [experiments](../configs/experiments/README.md) | Matching seasonal or meta runner |
| Separate searches | [tuning](../configs/tuning/README.md) | Matching Optuna script |
| Original step-controlled pipeline | [legacy](../configs/legacy/README.md) | `scripts.run_pipeline` |
| Predictor keep/drop and grouping | [featureconfig.yaml](../configs/features/featureconfig.yaml) | Feature selection helpers |
| Saved feature-importance aggregation | [postprocessing](../configs/postprocessing/README.md) | `scripts.analyze_feature_importance` |
| Historical winter hunting correction | [review-corrections.yaml](../review/config/review-corrections.yaml) | `review-corrections.py` |

[catalog.yaml](../configs/catalog.yaml) inventories maintained YAMLs, roles, runners and manuscript references. It is documentation metadata, not a modelling config.

## Schemas

**Seasonal model configs** contain `experiment`, `multitemporal`, `presence_data`, `predictors`, engines and outputs. Both temporal runners use the `multitemporal` key; the runner determines whether training years remain separate or are averaged. A filename or an invented `temporal` key cannot change the algorithm.

**Aggregate configs** contain `meta` and `scenarios`. Each scenario has a key, label, script and complete nested `config`. The meta runner reads those mappings and writes temporary resolved YAMLs. `meta.description` is descriptive, not an execution switch. Mentioning Optuna there does not trigger tuning.

**Legacy configs** contain pipeline step switches. Disabled steps can prevent execution even when their parameters are present.

**Review configs** use source aggregate selectors, patches, controls, validation and export settings. Review runs write immutable resolved copies.

Files intentionally remain self-contained. There is no general YAML `extends`, `include` or environment-variable expansion. Do not add these keys to remove repetition: the loaders would not apply them.

## Paths

| Context | Relative path base |
|---|---|
| Seasonal/meta/legacy model inputs and feature configs | Working directory, normally repository root |
| Review source aggregates | Repository root resolved by the launcher |
| Feature-importance analysis inputs/outputs | Directory containing the analysis YAML |

Moving a model YAML into `configs/` does not change the meaning of `data/...`, `outputs/...` or `exps/...`. For the analysis YAML only, `../../outputs/feature_importance_analysis` places output in the root's ignored `outputs/` directory.

Old root paths have moved. Internal defaults and review selectors were updated; external commands need the [migration map](CONFIG_MIGRATION.md). There are no compatibility symlinks. The two LOYO meta commands now default to the reviewed presets.

## Core settings

| Setting | Meaning |
|---|---|
| `experiment.name` | Run identifier; use a new name for a scientifically distinct run |
| `experiment.random_seed` | Configured seed, usually 42; archive with environment/randomness details |
| `run_test_year` | `all` or one outer test year; preserve the full training-year list |
| `multitemporal.years` | Available model years, including the held-out year |
| `multitemporal.seasons` | One season per aggregate scenario: `Winter` or `Summer` |
| `multitemporal.shift_km` | Synthetic raster layout offset; not an ecological buffer or CV block size |
| `multitemporal.include_hunting` | Allows hunting as a candidate; does not force retention after pruning |
| `multitemporal.background_per_year` | Background target per year, 15,000 in reviewed examples |
| `multitemporal.validation_fraction` | Within-training validation fraction, separate from outer LOYO |
| `multitemporal.stack_pattern` | Annual stack location; caches must match current inputs |
| `presence_data.file` / `.crs` | Prepared CSV and its actual coordinate reference system |
| `presence_data.thinning` | Environmental/grid thinning and source/year-specific settings |
| `bias_grid.pattern` | Annual/seasonal bias raster template; filename case matters |
| `background.bias_strength` | Sampling-weight control, differing among experiment families |
| `predictors.target_resolution_m` | Common modelling grid, 1,000 m in supplied examples |
| `predictors.variable_selection` | Correlation/VIF and reference-year settings |
| `multitemporal.vif` | Temporal-stack group limits and selection controls |
| `model.engines` | Engines for direct seasonal runs; comparison meta runners enforce both |
| `maxent.output_type` | MaxEnt output transformation, `cloglog` in study examples |
| `gbm.calibration` | Optional score calibration; consult supported methods in code |
| `variable_importance.permutation.n_repeats` | Repeats for the consuming routine; reviewed presets use five |
| `evaluation.cross_validation` | Additional CV; `method: spatial_block` does nothing when `enable: false` |

See [Predictors](PREDICTORS.md) for layers and names. Fold-specific selected-variable lists establish actual fitted features.

## Reviewed preset changes

Relative to their historical aggregate sources, the [reviewed presets](../configs/reviewed/README.md) enable hunting for winter in both temporal modes, retain summer exclusion, set five permutation repeats, add an `_reviewed` experiment-name suffix, and clarify descriptive metadata. Model parameters, thinning fractions, background choices and layer definitions are inherited. Historical files retain their scientific values so the correction package can compare source and patched settings.

These are study starting configurations, not recovered per-fold Optuna results.

## Editing a run

1. Copy a catalogued preset into an ignored experiment directory and record the intended scientific change.
2. Update the experiment name and the nested scenario `config`, not just `meta`.
3. When changing temporal coverage, update occurrence filters and all relevant layer-year lists.
4. Keep source filters and bias surfaces consistent with the research question.
5. Rebuild caches when candidate sets, grids, masks, years or inputs change.
6. Archive resolved YAML, feature config, selected variables, commit and environment.

Do not independently edit 44 legacy snapshots for a new experiment. Their [guide](../configs/legacy/README.md) explains scope and consumers.

## Validation

```bash
python scripts/validate_documentation.py
python scripts/validate_repository.py
```

Checks cover malformed YAML, duplicate keys, catalog coverage, config/script references, local Markdown links, core citation metadata and the reviewed scenario matrix. They do not access research data or certify archived model provenance.
