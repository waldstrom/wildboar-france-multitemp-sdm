# Configuration guide

The codebase grew through several experiment families. Configuration files are retained to preserve the analyses, while this guide identifies their intended role.

## Main entry configurations

| File | Intended use |
|---|---|
| `configwinter.yaml` | baseline/legacy Winter model |
| `configsummer.yaml` | baseline/legacy Summer model |
| `configall.yaml` | baseline model using all seasons |
| `configwinter_monotemporal.yaml` | seasonal monotemporal Winter workflow |
| `configsummer_monotemporal.yaml` | seasonal monotemporal Summer workflow |
| `configmultitemporal_winter.yaml` | multitemporal Winter workflow |
| `configmultitemporal_summer.yaml` | multitemporal Summer workflow |
| `config_gbm_loyo_compare.yaml` | aggregate leave-one-year-out comparison scenarios |
| `configtuning.yaml` and `*optuna*.yaml` | parameter-search experiments |
| `config_gbm*.yaml` | GBM/MaxEnt comparison experiments |
| `featureconfig.yaml` | predictor grouping, inclusion and exclusion rules |

The explicit year/source matrix below `configs/` is described in [`../configs/README.md`](../configs/README.md).

## Paths

Paths are resolved relative to the repository root in the supplied examples. Input paths point below `data/`, which is intentionally empty except for acquisition guides. Generated products point below `outputs/`, `exps/`, `processed_data/` or `correlation/`; all are ignored by Git.

Before a run:

1. Read [`../data/README.md`](../data/README.md) and the relevant subdirectory guides.
2. Place authorised, prepared inputs at the filenames referenced by the selected YAML file, or edit the YAML paths.
3. Confirm the occurrence coordinate reference system rather than relying on column names.
4. Confirm that every raster shares the configured CRS, extent, resolution, origin and nodata convention.
5. Use a new experiment name/output directory rather than overwriting an earlier run.

## Seasonal conventions

Several workflows distinguish `Summer` and `Winter`, and winter may cross calendar-year boundaries. Preserve the exact model-year assignment used when preparing occurrence records, ERA5-Land summaries, bias maps and hunting-bag covariates. The YAML labels alone do not reconstruct that decision.

## Running

```bash
python scripts/run_pipeline.py --config configwinter.yaml
python scripts/run_pipeline.py --config configsummer.yaml
python scripts/meta_run_leave_one_year.py --config config_gbm_loyo_compare.yaml
```

Some specialised scripts expose different command-line arguments. Use `python <script> --help` where available and consult the script docstring before launching a large run.

## Adding a new configuration

Keep paths relative, avoid machine-specific directories, set a reproducible random seed, and document the scientific difference from the closest existing configuration. Never place access tokens, credentials or confidential data locations in YAML committed to Git.