# Configuration index

[Repository home](../README.md) · [Configuration reference](../docs/CONFIGURATION.md) · [Path migration](../docs/CONFIG_MIGRATION.md)

Start with `reviewed/` for the corrected study examples. Other folders retain useful historical experiments and their original scientific settings.

| Directory | Role | Manuscript links |
|---|---|---|
| [reviewed](reviewed/README.md) | Two corrected LOYO starting configurations | S5; Table 3; S6a; S8 |
| [experiments](experiments/README.md) | Historical seasonal/engine comparison presets | S4-S5; pre-correction provenance |
| [tuning](tuning/README.md) | Separate search experiments, with explicit objective caveats | S5 |
| [legacy](legacy/README.md) | Original pipeline and 44 year/source snapshots | S1-S4; historical comparisons |
| [features](features/README.md) | Candidate rules and groups | S4 |
| [postprocessing](postprocessing/README.md) | Saved-importance analysis input template | S6b support |
| [review/config](../review/config) | Correction orchestration, overlay and export manifest | Review-stage audit |

[catalog.yaml](catalog.yaml) is the complete machine-readable inventory, including each runner, previous path and study references. Run `python scripts/validate_documentation.py` from the root after changes. The catalog itself is not passed to a modelling runner.

Model paths resolve from the repository root even though configs are in subdirectories. The postprocessing YAML is the documented exception: its paths resolve from its own directory.
