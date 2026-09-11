# Contributing

[Documentation](docs/README.md) · [Configuration guide](docs/CONFIGURATION.md) · [Licence](LICENSE)

Keep changes focused and scientifically traceable. Identify the affected runner/configuration and distinguish documentation or refactoring from changes to fitting, sampling, evaluation or predictor semantics.

## Code and configuration changes

- Use repository-relative model paths and preserve deterministic controls.
- Add new YAMLs to [configs/catalog.yaml](configs/catalog.yaml), including their consumer and study references.
- Put reviewed starting examples, historical experiments, tuning and legacy configs in their documented folders.
- Do not silently change historical settings or treat a candidate as a guaranteed retained feature.
- Update [the manuscript crosswalk](docs/MANUSCRIPT_CROSSWALK.md) when a publication mapping changes.
- Preserve existing output paths unless the change includes a migration plan for saved-run consumers.
- Use small synthetic inputs for meaningful regression tests; do not bundle research data as fixtures.

## Data policy

Do not commit occurrence/collision records, environmental rasters, vectors, fitted models, result tables, figures, archives, credentials or confidential local manifests. Add acquisition/preparation documentation when introducing a new data category. The [data guide](data/README.md) defines source access and grid conventions.

Generated products belong below ignored experiment/output directories. Document inputs, software versions and transformations in authorized local run records.

## Checks

```bash
python scripts/validate_repository.py
python scripts/validate_documentation.py
python -m compileall -q scripts review review-corrections.py review-finalize-existing-run.py
```

For relevant code changes, install development requirements and run the affected synthetic tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Lightweight CI validates the code-only tree, YAML/catalog, documentation links and metadata and compiles Python. It does not fetch input data or fit full models.

## Pull requests and releases

Describe the problem, change, affected workflows, scientific parameter changes (if any) and verification performed. Link affected configs and document any reproduction limitation. The [release guide](docs/PUBLIC_RELEASE.md) covers code-only exports and metadata updates.
