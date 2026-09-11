# Data directory — intentionally empty

No research data are distributed with this repository. Only `README.md` files are tracked below `data/`; all tabular, vector, raster, array, model-input and intermediate files must remain local.

This separation is deliberate because the workflow combines open environmental products with occurrence records that have different licences and access conditions. In particular, SNCF and RRN records must not be redistributed without permission from their data holders.

## Expected local layout

After obtaining and preparing the inputs, the local directory should resemble:

```text
data/
├── allpoints.csv
├── boundary_france.asc
├── boundary_france.prj
├── regions_2154.gpkg
├── france-neighbours.gpkg
├── 10_largest_cities_2154.gpkg
├── bioclim/
├── dem/
├── era5/
├── hunting-bag/
├── land-cover/
├── lc-distances/
├── sat-indices/
├── special/
├── theia-special/
├── bias-maps/       # derived
└── stacks/          # derived
```

Only the files needed by the selected YAML configuration are required. Each subdirectory contains a separate acquisition and preparation guide.

## Occurrence records: `allpoints.csv`

The combined table contains one wild-boar observation or collision record per row. For compatibility across the legacy and multitemporal workflows, retain the following fields where available:

| Field | Meaning |
|---|---|
| `dd long`, `dd lat` | coordinate columns used by the legacy pipeline |
| `year`, `year-int` | Model-season year; Winter uses its starting year. Configurations use different column names |
| `month`, `month-int` | month label and integer month |
| `source` | `GBIF`, `RRN` or `SNCF` |
| `season` | `Summer` or `Winter` |
| `semester` | legacy seasonal field |

**Coordinate check:** the historical column names suggest longitude/latitude, while several configurations declare `EPSG:2154`. Set `presence_data.crs` to the actual coordinate reference system of your local table; never relabel coordinates without transforming them. This must be resolved and documented before a reproducible release.

### GBIF

Create a citable GBIF occurrence download for *Sus scrofa* in France with coordinates and the required date range. Save the GBIF download DOI, query, access date and licence information in your local data manifest. Clean obvious coordinate and taxonomic issues, then map the result into the schema above.

- Portal: <https://www.gbif.org/>
- Data-use guidance: <https://www.gbif.org/citation-guidelines>

### RRN and SNCF

These structured roadkill and railway-collision records are not public inputs of this repository. Request access from the respective data holders under the original project agreements. Keep raw extracts outside the repository, remove direct identifiers if present, and document every filtering and harmonisation step. The code can still be inspected and run with authorised local copies or with a compatible independently collected dataset.

## Study-area support files

Administrative boundaries can be reconstructed from the official IGN ADMIN EXPRESS product: <https://geoservices.ign.fr/adminexpress>. Reproject to Lambert-93 (`EPSG:2154`), select metropolitan mainland France according to the study definition, and derive the regional/neighbour/city layers required by the chosen analysis.

## Common grid contract

Unless a configuration explicitly states otherwise, prepared rasters should use:

- CRS: Lambert-93 (`EPSG:2154`)
- cell size: 1,000 m
- identical extent, origin, dimensions and pixel alignment
- nodata value: `-9999`
- mainland-France mask consistent across all layers

Continuous variables should normally be aggregated/resampled continuously; class labels should use nearest-neighbour resampling. Fractional cover layers must retain their documented scale (`0–1` for the annual project layers; some THEIA-derived inputs use `0–100` before harmonisation).

## Local provenance record

For every input, record at minimum: provider, product name, release/version, DOI or download URL, access date, licence, original CRS/resolution, processing command or script, output checksum and any access restriction. Do not commit that manifest if it contains confidential paths, credentials or restricted metadata.

## Repository rule

Run `python scripts/validate_repository.py` before every commit. It fails when data or generated outputs are tracked below this directory.
## Links to the reviewed study

Winter `y` means September `y` through February `y+1`; Summer `y` means March-August `y`. Thus Winter 2023 requires records and seasonal inputs extending into February 2024. Prepare those labels upstream; the seasonal loader filters supplied labels.

See [Predictors](../docs/PREDICTORS.md) for units and feature naming, [Configuration](../docs/CONFIGURATION.md) for path semantics, and [the manuscript crosswalk](../docs/MANUSCRIPT_CROSSWALK.md) for S1-S5 links.
