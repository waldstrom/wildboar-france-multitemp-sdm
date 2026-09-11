# Predictor and data reference

[Documentation home](README.md) · [Configuration](CONFIGURATION.md) · [Data acquisition](../data/README.md)

## Six domains in Fig. 4

The six domains in the manuscript organize ecological information. They are not a one-to-one mapping to YAML keys: for example, climate inputs are split between `bioclim` and `era5`, while several auxiliary products share `special` or `theia_special`.

| Manuscript domain | YAML/data families | Preparation and reference |
|---|---|---|
| Land-cover fractions | `landcover_fraction`; selected `theia_special` products | [Annual land cover](../data/land-cover/README.md), [THEIA](../data/theia-special/README.md); S3-S4 |
| Land-cover distances | `landcover_distance` | [LC distances](../data/lc-distances/README.md); S4 |
| Spectral indices | `veg_index` | [Satellite indices](../data/sat-indices/README.md); S4 |
| Climate | `bioclim`, `era5` | [BIOCLIM](../data/bioclim/README.md), [ERA5](../data/era5/README.md); S4 |
| Terrain | `dem` | [DEM](../data/dem/README.md); S4 |
| Auxiliary / anthropogenic and landscape structure | `hunting_bag`, `special`, selected `theia_special` products | [Hunting](../data/hunting-bag/README.md), [special layers](../data/special/README.md), [THEIA](../data/theia-special/README.md); S4 |

## File contracts

| Family | Typical configured pattern | Check before fitting |
|---|---|---|
| Annual cover | `data/land-cover/{year}_class{class}.asc` | Class identifiers and fraction scale |
| Distance | `data/lc-distances/{year}_class{class}_distance.asc` | Distance definition, units and matching class/year |
| Spectral indices | `data/sat-indices/{index}/{year}_{month}_{index_lc}.tif` | Exact band formula, acquisition/aggregation period and scaling |
| Seasonal weather | `data/era5/*_{year}_{season}_*.asc` | Winter-year convention, statistic, unit and filenames |
| Hunting | `data/hunting-bag/hunting_bag_{year}.asc` | Season/statistical year, denominator and source provenance |
| Static layers | Explicit file lists under `bioclim`, `dem`, `special`, `theia_special` | Product release, transformation and units |
| Sampling bias | `data/bias-maps/...{year}...{season}....asc` | Occurrence-source scenario, season/year and weight scale |

All modelling rasters must share the actual grid transform, CRS, resolution, extent, shape and mask. EPSG:2154 plus a 1,000 m cell size is insufficient to guarantee pixel alignment. The examples use nodata `-9999`; preserve valid zeros separately from nodata.

Annual project fractions are documented on a 0-1 scale; some THEIA-derived fractions use 0-100 before harmonization. Record conversions rather than assuming all similarly named layers share a scale. Physical units of ERA5 derivatives must be documented per variable and preprocessing statistic.

The reviewed supplement itself flags NDSI abbreviation ambiguity. This SDM repository consumes precomputed index rasters. A filename does not resolve snow versus salinity formulations; record the actual upstream band formula and scale. Do not relabel or regenerate an index from an ambiguous acronym.

## Candidate versus retained predictor

1. `predictors.layers` lists available candidate files/patterns.
2. Seasonal/year expansion and `include_hunting` determine candidates admitted to the stack.
3. [featureconfig.yaml](../configs/features/featureconfig.yaml) supplies explicit variable rules and groups; temporal fusion can harmonize year-specific names.
4. Correlation/VIF processing and the configured group constraints reduce redundancy.
5. The saved selected-predictor list and model metadata establish the variables actually fitted, including order.

`keep` permits a candidate through that rule stage; it is not a guarantee of final retention. Conversely, the presence of an annual file does not guarantee it was used. Check `ignore_featureconfig_drop` and the relevant consumer when comparing experiments.

## Names in importance tables

Names can be transformed during year fusion and layer loading. For instance, an annual `2019_class10_BEECHGRV_distance` variable can be represented by its harmonized class/distance name in a temporal model. Match saved names to the exact model before combining importance tables; do not join by approximate text alone.

The manuscript distinguishes land-cover fractions, distances to classes, fragmentation density and small woody features. These are distinct predictors. A dense-urbanization raster name does not by itself justify a statement about peri-urban greenbelts or refuge habitat. Tables 4-5 and S6b describe predictive associations for retained variables.

## Source records

Use [DATA_PROVENANCE_TEMPLATE.md](DATA_PROVENANCE_TEMPLATE.md) for provider, release, access date, licence, CRS, native resolution, unit, processing command and checksum. Include calendar-to-model-year conversion and candidate/selection history. Local provenance that contains restricted locations or access details stays outside version control.
