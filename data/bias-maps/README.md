# Bias maps — derived, not distributed

This directory is reserved for sampling-bias surfaces. These files are generated from authorised occurrence data and/or openly available covariates; they are not primary downloads and must not be committed.

Expected names include:

```text
Bias-Map-GBIF.asc
Bias-Map-RRN.asc
Bias-Map-SNCF.asc
Combined_2017-Winter.asc
Combined_2018-Summer.asc
...
```

The exact files are controlled by `bias_grid.file` or `bias_grid.pattern` in the selected YAML configuration. Recreate them with the pipeline's `bias_grid` stage (`occurrence_density`, `lc_urban` or `predefined`, depending on the experiment). Preserve the target-grid alignment, CRS, nodata value, kernel/bandwidth settings and combination rule in the run metadata.

Because a bias surface can reveal the spatial distribution of restricted SNCF/RRN observations, treat derived maps from those sources as restricted unless the data holder has approved release.