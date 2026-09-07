# Contributing

Contributions should keep this repository inspectable, reproducible and safe to share.

## Before changing code

Open a focused branch and identify the configuration and workflow affected by the change. Avoid mixing scientific-method changes, refactoring and regenerated result files in one commit.

## Data policy

This is a code-only repository. Do not commit occurrence records, collision records, environmental rasters, vectors, model objects, figures, result tables, archives or local provenance files. Tests should create small synthetic inputs at runtime in temporary directories rather than add fixtures containing research data.

When code introduces a new path below `data/<category>/`, add or update `data/<category>/README.md` with:

- the expected filenames and schema;
- the authoritative provider or access route;
- licence and access restrictions;
- spatial, temporal and unit conventions;
- the processing needed to reach the pipeline-ready format.

Never commit credentials, private download links, confidential agreement text, personal identifiers or machine-specific absolute paths.

## Code and configuration changes

Keep paths relative to the repository root. Preserve deterministic seeds where randomness is used. Add a concise docstring or comment for non-obvious scientific transformations, and update `docs/CONFIGURATION.md` when entry points or configuration semantics change.

Generated scenario files under `configs/` should be regenerated from their source logic rather than edited inconsistently one by one.

## Checks

Run the lightweight release checks before opening a pull request:

```bash
python scripts/validate_repository.py
python -m compileall -q scripts review review-corrections.py review-finalize-existing-run.py
```

For the full test suite, install the development requirements and run:

```bash
python -m pip install -r requirements-dev.txt
pytest
```

Full geospatial tests may require system libraries compatible with Rasterio, GeoPandas and LightGBM.

## Public releases

Do not make the historical private repository public. Follow `docs/PUBLIC_RELEASE.md` to create a new history-free repository after the publication tree has been reviewed.