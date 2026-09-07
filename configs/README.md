# Generated scenario configurations

This directory contains the explicit year × season × observation-source configurations used by the legacy scenario runner. They are text-only reproducibility records, not input data.

Naming convention:

```text
<year>-<season>_<source>.yaml
```

where `<source>` is `ALL`, `GBIF`, `RRN` or `SNCF`.

The files preserve the exact paths and parameter values expected by the pipeline. They deliberately refer to local files below `data/`; those files are not distributed. See [`../data/README.md`](../data/README.md) before running a scenario.

Do not edit dozens of generated files independently. Change the appropriate root configuration or generation logic, regenerate the matrix, and review the resulting diff. A scenario can be run with:

```bash
python scripts/run_pipeline.py --config configs/2022-Winter_ALL.yaml
```

The repository hygiene check permits YAML files here but rejects tracked data and generated outputs.