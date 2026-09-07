# Bioclimatic variables

The configurations expect WorldClim-style BIO variables such as `bio01-ann_mean_temp.asc` through `bio19_precip_coldest_q.asc`.

A reproducible public source is WorldClim version 2.1: <https://www.worldclim.org/data/worldclim21.html>. Confirm the release, baseline period and source resolution used in the manuscript before reproducing final results. Download the 19 bioclimatic variables, crop to the study area, reproject to `EPSG:2154`, align to the common 1-km grid and write the filenames referenced by the active configuration.

Document:

- WorldClim version and baseline period
- source resolution
- download/access date
- resampling and aggregation method
- clipping mask and output checksums

These layers are static covariates and should not be committed.