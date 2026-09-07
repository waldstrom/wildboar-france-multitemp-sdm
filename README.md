# Wild Boar Habitat Suitability Modelling for France

This repository contains the Python workflow used to prepare, fit and evaluate seasonal wild-boar habitat-suitability models for mainland France. It includes MaxEnt modelling through `elapid`, optional GBM comparisons, predictor selection, occurrence thinning, bias-aware background sampling, leave-one-year-out experiments, evaluation, novelty analysis and feature-importance utilities.

> **Code-only release:** research data and generated results are deliberately not distributed here. Every expected data location contains a `README.md` describing the required format, source or access route, and preparation steps.

## What is included

- modelling, preprocessing, evaluation and visualisation code;
- YAML configurations for baseline, seasonal, monotemporal, multitemporal, Optuna and GBM-comparison experiments;
- explicit year × season × data-stream scenario configurations;
- tests and a repository-hygiene validator;
- manuscript-review utilities under `review/`;
- data-acquisition and provenance documentation.

The repository does **not** include GBIF exports, RRN or SNCF records, environmental rasters, administrative layers, hunting-bag tables, predictor stacks, model objects, figures or result tables. Some source products are openly obtainable; RRN and SNCF records require authorisation from their respective data holders. Exact numerical reproduction also requires matching the source releases, preprocessing decisions and software environment used for the analysis.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/` | core pipeline, model training, evaluation and experiment runners |
| `configs/` | generated year/season/source scenario configurations |
| `data/` | documentation-only placeholders for local inputs |
| `outputs/` | documentation-only placeholder for generated results |
| `review/` | code used to audit and rebuild review-stage artefacts |
| `tests/`, `review/tests/` | automated tests |
| `docs/` | configuration, provenance and public-release guidance |

## Installation

Python 3.10 or newer is required by the type syntax used in the code. Create an isolated environment and install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. The convenience scripts `setup.sh` and `setup.ps1` install the same runtime requirements. Geospatial packages may additionally require platform-compatible GDAL/PROJ libraries.

`requirements.txt` records the required packages but does not pin a complete historical environment. For an archival analysis, record the exact resolved versions alongside the run provenance.

## Prepare the data

Start with [`data/README.md`](data/README.md), then read the guide in each required subdirectory. The supplied configurations expect local paths below `data/`, but no downloader silently fetches or redistributes restricted material.

Before running a model, verify that:

1. the occurrence table follows the documented schema and its declared CRS matches the actual coordinates;
2. all rasters share the configured CRS, extent, origin, dimensions, resolution and nodata convention;
3. Summer/Winter labels and cross-calendar winter years are assigned consistently across occurrences and predictors;
4. every input has a local provenance record, ideally based on [`docs/DATA_PROVENANCE_TEMPLATE.md`](docs/DATA_PROVENANCE_TEMPLATE.md).

## Select a configuration

The main entry configurations are summarised in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md). Typical examples are:

```bash
# Baseline seasonal models
python scripts/run_pipeline.py --config configwinter.yaml
python scripts/run_pipeline.py --config configsummer.yaml

# Aggregate leave-one-year-out comparison
python scripts/meta_run_leave_one_year.py --config config_gbm_loyo_compare.yaml

# One explicit year/source scenario
python scripts/run_pipeline.py --config configs/2022-Winter_ALL.yaml
```

Use `python <script> --help` for specialised runners that expose command-line options. Large runs create experiment-specific output directories; do not reuse a directory unless the workflow explicitly supports resuming it.

## Review-stage utilities

The `review/` package and the two root-level review scripts reconstruct review tables, figures and diagnostics from authorised local run artefacts. `review/input/` and `review/output/` contain documentation only. See [`review/README.md`](review/README.md) for the expected package structure and commands.

## Validation and tests

The hygiene check uses only the Python standard library and should run before every commit:

```bash
python scripts/validate_repository.py
python -m compileall -q scripts review review-corrections.py review-finalize-existing-run.py
```

Install development dependencies to run the full test suite:

```bash
python -m pip install -r requirements-dev.txt
pytest
```

Continuous integration performs the data-free repository check and compiles all Python sources. It intentionally does not download data or launch computationally expensive model runs.

## Outputs

Generated models, maps, figures, logs, response curves, statistics and cached stacks belong in ignored local directories. See [`outputs/README.md`](outputs/README.md) for archiving guidance.

## Citation

If you use this code or workflow in scientific work, please cite the associated preprint:

Meyer AF, Reibel T, Morelle K, Kneubühler M, Jordan D (2026) Wild Boar Collision Data and Satellite Computer Vision Refine Habitat Suitability Mapping across France. Research Square, Version 1. https://doi.org/10.21203/rs.3.rs-8798859/v1

The preprint is available at Research Square and is licensed separately under CC BY 4.0.

For software-specific citation metadata, see CITATION.cff. When citing a specific archived software release, please additionally cite the corresponding release DOI if available.

Copyright © 2026 Adrian Ferdinand Meyer and contributors.

## Licence 

The source code in this repository is released under the BSD 3-Clause License. See LICENSE for the full licence terms.

Unless explicitly stated otherwise, this licence applies to the software, configuration files and original documentation contained in this repository. It does not grant rights to third-party datasets or other externally sourced materials. GBIF, RRN, SNCF, environmental, administrative and other input datasets remain subject to the licences, terms of use and access restrictions imposed by their respective data providers.

No restricted research data are distributed with this repository.

If you use this software in scientific work, please cite the associated publication and, where applicable, the archived software release. Citation information will be provided in CITATION.cff.

Copyright © 2026 Adrian Meyer and contributors.
