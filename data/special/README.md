# Additional static predictors

Configurations reference additional project-specific layers such as:

```text
Zones_humides_INPN.asc
fragmentation_density.asc
small_woody_features.asc
wildareas_humanfootprint.asc
```

Public starting points include:

- French nature-reference downloads / wetlands: <https://inpn.mnhn.fr/telechargement/cartes-et-information-geographique>
- Copernicus Small Woody Features: <https://land.copernicus.eu/en/products/high-resolution-layer-small-landscape-features/small-woody-features-2018>

`fragmentation_density.asc` and `wildareas_humanfootprint.asc` are derived products whose exact upstream version, algorithm and transformation are not recoverable from their filenames alone. Before a public reproducibility claim, identify and document the original source, licence, year, formula, moving-window size, normalisation and grid conversion. Until then, treat these as user-supplied covariates and clearly label any substitute as non-identical.

Prepare all layers on the common grid and do not commit them.