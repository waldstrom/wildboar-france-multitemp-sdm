# Legacy pipeline configurations

[Config index](../README.md) · [Reviewed study configs](../reviewed/README.md)

| File / directory | Scope |
|---|---|
| [config.yaml](config.yaml) | Original general-purpose pipeline preset |
| [configall.yaml](configall.yaml) | Original all-season preset |
| [configwinter.yaml](configwinter.yaml) | Original winter preset |
| [configsummer.yaml](configsummer.yaml) | Original summer preset |
| [scenarios](scenarios) | 44 explicit year × season × source snapshots |

Run a legacy preset with `python -m scripts.run_pipeline --config configs/legacy/configwinter.yaml`. Pipeline `steps` switches determine which processing stages execute.

The scenario files cover Winter 2017-2022 and Summer 2018-2022, each with ALL, GBIF, RRN and SNCF variants. They are historical records, not the final 2017-2023/2018-2023 LOYO matrix. Basenames and parsed scientific values are retained.

`python -m scripts.seasontester --dry` prepares isolated local run copies without executing models. The sequential and parallel season testers consume only `configs/legacy/scenarios/`. They do not regenerate canonical snapshots from a single maintained source template, so do not infer an available regeneration command from the old 'generated configs' description.

For new study runs, start from the reviewed presets instead of editing dozens of snapshots independently.
