# Reviewed study starting configurations

[Config index](../README.md) · [Methods](../../docs/METHODS.md) · [Reproduction status](../../docs/REPRODUCIBILITY.md)

| File | Occurrence scenario | Run command from repository root |
|---|---|---|
| [loyo_all_sources.yaml](loyo_all_sources.yaml) | GBIF + RRN + SNCF | `python -m scripts.meta_run_leave_one_year --config configs/reviewed/loyo_all_sources.yaml` |
| [loyo_gbif.yaml](loyo_gbif.yaml) | GBIF only | `python -m scripts.meta_run_leave_one_year_gbif --config configs/reviewed/loyo_gbif.yaml` |

Each file embeds winter/summer × monotemporal/multitemporal scenarios. Both meta runners enforce MaxEnt (`elapid`) and GBM. Winter model years are 2017-2023, summer 2018-2023; the outer test is temporal LOYO.

These files derive from the historical all-source and GBIF LOYO aggregates. They enable hunting as a candidate in **both winter modes**, retain summer exclusion, set **five configured permutation repeats**, and add `_reviewed` to experiment names. Every other scientific setting is inherited. Permitted candidates can still be removed by pruning.

These are explicit, runnable starting configurations once inputs exist. They do not contain recovered fold-specific Optuna outcomes and do not automatically execute nested tuning. Setting five repeats does not change a helper's evaluation split or make a best-fold export an all-fold S6b table. See the [implementation differences](../../docs/REPRODUCIBILITY.md).

Use these for new study-aligned runs. Use the [historical sources](../experiments/README.md) and [review correction package](../../review/README.md) when inspecting original-versus-corrected provenance. Extract an inner scenario for a direct single-engine/seasonal run as shown in the [quick start](../../docs/QUICKSTART.md#run-one-seasonal-scenario-first).
