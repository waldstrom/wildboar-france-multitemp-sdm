# Wild boar habitat suitability across France

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Study preprint](https://img.shields.io/badge/Preprint-Research_Square-008080.svg)](https://doi.org/10.21203/rs.3.rs-8798859/v1)

Python research workflows for **seasonal, presence-background habitat suitability modelling** of *Sus scrofa* in mainland France. The study combines GBIF observations, road and rail wildlife-vehicle collisions, and annually updated environmental predictors to compare **multitemporal and monotemporal MaxEnt and LightGBM models** using leave-one-year-out evaluation.

Companion code to **Wild Boar Collision Data and Satellite Computer Vision Refine Habitat Suitability Mapping across France**, Adrian Ferdinand Meyer, Théo Reibel, Kevin Morelle, Mathias Kneubühler and Denis Jordan (2026).

> This is a code-only research repository. Input data, fitted models and publication outputs are obtained separately. The reviewed presets document corrected candidate settings, but are not an archive of fitted models or fold-specific tuning results. See the [reproduction status](docs/REPRODUCIBILITY.md).

## Start here

| Your task | Guide |
|---|---|
| Install the environment and run the first checks | [Quick start](docs/QUICKSTART.md) |
| Choose or adapt a configuration | [Configuration guide](docs/CONFIGURATION.md) and [config index](configs/README.md) |
| Find code behind a manuscript figure, table or supplement | [Manuscript crosswalk](docs/MANUSCRIPT_CROSSWALK.md) |
| Understand the study design and implementation | [Methods and workflow](docs/METHODS.md) |
| Prepare occurrences, rasters and bias layers | [Data guide](data/README.md) and [predictor reference](docs/PREDICTORS.md) |
| Rebuild review-stage outputs from existing runs | [Review workflow](review/README.md) and [output reference](docs/OUTPUTS.md) |
| Reuse, contribute to or archive the software | [Contributing](CONTRIBUTING.md), [citation](CITATION.cff), [release guide](docs/PUBLIC_RELEASE.md) |

The [documentation home](docs/README.md) links all guides. The [configuration migration table](docs/CONFIG_MIGRATION.md) maps previous filenames to their new locations.

## Study at a glance

| Dimension | Definition |
|---|---|
| Domain | Mainland France; common SDM grid in Lambert-93, EPSG:2154, at 1 km resolution |
| Summer | March to August, study model years 2018 through 2023 |
| Winter | September to February, labelled by its starting year, 2017 through 2023 |
| Occurrence scenarios | GBIF only; GBIF + RRN road collisions + SNCF rail collisions |
| Temporal representations | Year-specific predictors; predictors averaged over training years |
| Model engines | MaxEnt through `elapid`; GBM through `lightgbm` |
| Outer evaluation | Leave one year out, then project to that year's predictor stack |
| Interpretation | Relative habitat suitability; not calibrated occurrence or collision probability |

Winter 2023 includes January and February 2024. A common 1 km modelling grid does not imply that every source product was observed at 1 km resolution. The land-cover classification stage uses 30 m imagery; its training assets are outside this snapshot.

## Install and inspect

Run from the repository root. Python 3.10+ is required by the source syntax; CI uses Python 3.11 for lightweight checks.

```bash
git clone https://github.com/waldstrom/wildboar-france-multitemp-sdm.git
cd wildboar-france-multitemp-sdm
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/validate_repository.py
python scripts/validate_documentation.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. See the [environment notes](docs/QUICKSTART.md#environment) for geospatial dependencies. The dependency list is not a historical lockfile.

After preparing the inputs and reading the [reviewed preset notes](configs/reviewed/README.md):

```bash
# Four seasonal/temporal scenarios per file, each run with both engines.
python -m scripts.meta_run_leave_one_year --config configs/reviewed/loyo_all_sources.yaml
python -m scripts.meta_run_leave_one_year_gbif --config configs/reviewed/loyo_gbif.yaml
```

These are full modelling jobs requiring local data and substantial computation. They use the supplied parameters; the meta runners do not launch nested Optuna tuning.

## Repository layout

| Directory | Contents |
|---|---|
| [configs/reviewed](configs/reviewed/README.md) | Corrected study starting configurations |
| [configs/experiments](configs/experiments/README.md) | Historical seasonal and engine comparisons |
| [configs/tuning](configs/tuning/README.md) | Separate parameter-search experiments |
| [configs/legacy](configs/legacy/README.md) | Earlier pipeline presets and 44 year/source snapshots |
| [configs/features](configs/features/README.md), [configs/postprocessing](configs/postprocessing/README.md) | Predictor selection and analysis settings |
| [scripts](scripts/README.md) | Processing, training, diagnostics and validation |
| [data](data/README.md), [outputs](outputs/README.md) | Input acquisition guides and local output locations |
| [review](review/README.md) | Hunting correction, saved-run finalization and audits |
| [docs](docs/README.md) | Methods, cross-references and reproducibility guidance |
| [tests](tests), [review/tests](review/tests) | Synthetic and unit tests |

## Citation and licence

Meyer AF, Reibel T, Morelle K, Kneubühler M, Jordan D (2026) Wild Boar Collision Data and Satellite Computer Vision Refine Habitat Suitability Mapping across France. Research Square, Version 1. [doi:10.21203/rs.3.rs-8798859/v1](https://doi.org/10.21203/rs.3.rs-8798859/v1).

[CITATION.cff](CITATION.cff) supplies software and preferred study citation metadata; [codemeta.json](codemeta.json) supplies machine-readable software metadata. Record the commit or archived software release used alongside the paper citation. The preprint DOI identifies the paper, not a software release.

Code, configuration files and original repository documentation are licensed under the [BSD 3-Clause License](LICENSE). Third-party data, model weights and publications retain their own licences and access conditions. See [licensing and attribution](docs/LICENSING.md).

Maintainer: Adrian Ferdinand Meyer, Institute Geomatics, FHNW. Code questions: [GitHub issues](https://github.com/waldstrom/wildboar-france-multitemp-sdm/issues). Data requests follow the [data guide](data/README.md).
