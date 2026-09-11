# Historical seasonal and comparison presets

[Config index](../README.md) · [Current starting points](../reviewed/README.md)

These files preserve the previous public snapshot's scientific settings. In particular, winter monotemporal presets can still have `include_hunting: false`. They are historical sources for the correction package, not the recommended corrected study entry points.

| Configuration | Runner | Purpose |
|---|---|---|
| [config_gbm-compare.yaml](config_gbm-compare.yaml) | `scripts.meta_run` | Fixed-year engine comparison |
| [config_gbm_loyo_compare.yaml](config_gbm_loyo_compare.yaml) | `scripts.meta_run_leave_one_year` | Historical all-source LOYO aggregate |
| [config_gbm_loyo_gbif.yaml](config_gbm_loyo_gbif.yaml) | `scripts.meta_run_leave_one_year_gbif` | Historical GBIF LOYO aggregate |
| [config_gbm_loyo_gbif_grid.yaml](config_gbm_loyo_gbif_grid.yaml) | `scripts.meta_run_leave_one_year_gbif` | GBIF grid-thinning sensitivity variant |
| [configmultitemporal_winter.yaml](configmultitemporal_winter.yaml) | `scripts.multitemporal` | Direct winter multitemporal run |
| [configmultitemporal_summer.yaml](configmultitemporal_summer.yaml) | `scripts.multitemporal` | Direct summer multitemporal run |
| [configwinter_monotemporal.yaml](configwinter_monotemporal.yaml) | `scripts.monotemporal` | Direct historical winter baseline |
| [configsummer_monotemporal.yaml](configsummer_monotemporal.yaml) | `scripts.monotemporal` | Direct summer baseline |

All commands use `python -m <runner> --config <path>` from the root. These configs are fully specified independently; identical-looking fields can differ among historical experiments. No scientific harmonization was applied during relocation.
