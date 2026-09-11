# Review correction and saved-run finalization

[Repository home](../README.md) · [Manuscript crosswalk](../docs/MANUSCRIPT_CROSSWALK.md) · [Reproduction status](../docs/REPRODUCIBILITY.md)

The review package reconstructs affected winter-model outputs from authorized local inputs and records the hunting-candidate correction. It is already integrated into this repository. The root launchers are [review-corrections.py](../review-corrections.py) and [review-finalize-existing-run.py](../review-finalize-existing-run.py).

## Original setting and correction

Historical winter monotemporal configurations disabled `multitemporal.include_hunting`, while matched multitemporal configurations enabled it. The correction runner selects those historical aggregate scenarios, writes immutable copies with hunting activated, and rebuilds their stacks/models. Historical source files stay available under [configs/experiments](../configs/experiments/README.md).

The [reviewed presets](../configs/reviewed/README.md) are separate starting examples for new runs. The correction package intentionally continues to use historical sources so its before/after audit remains meaningful.

Candidate inclusion is not forced retention. Audits distinguish annual hunting-raster availability, inclusion in the master stack, inclusion in each materialized fold, and retention after predictor selection.

## Configuration files

| File | Purpose |
|---|---|
| [review-corrections.yaml](config/review-corrections.yaml) | Scenarios, historical source selectors, patches, controls and validation |
| [configwinter_monotemporal_hunting.overlay.yaml](config/configwinter_monotemporal_hunting.overlay.yaml) | Minimal hunting-candidate correction |
| [artifact-manifest.yaml](config/artifact-manifest.yaml) | Stable export directories plus current manuscript references |

The default model set comprises corrected winter monotemporal and matched multitemporal reference runs for all-source and GBIF-only data, using both engines. Summer is outside this correction. Optional no-hunting controls reproduce the historical candidate setting.

## Inspect, fit and resume

```bash
python review-corrections.py self-test
python review-corrections.py plan
python review-corrections.py run --dry-run
```

The dry run writes resolved local configurations and input/audit information without model fitting. It still needs the configured software environment and can report missing research inputs.

After supplying those inputs:

```bash
python review-corrections.py run
python review-corrections.py run --resume review/output/EXISTING_RUN
```

Replace `EXISTING_RUN` with the actual run directory. Completed work is reused according to the runner's checks; incomplete artifacts are handled in isolated run directories.

| Option / command | Effect |
|---|---|
| `run --corrected-only` | Limits fitting to corrected monotemporal scenarios; comparison products may lack references |
| `run --include-controls` | Adds explicit historical no-hunting controls |
| `export --run-dir ...` | Rebuilds exports from an available run |
| `validate --run-dir ...` | Checks an existing run against package requirements |

## Finalize existing results

```bash
python review-finalize-existing-run.py --source-run review/output/EXISTING_RUN --n-repeats 5
```

Finalization does not refit models. It reads saved metrics/configuration audits, rebuilds summaries and can compute an all-fold hunting permutation diagnostic for corrected winter monotemporal MaxEnt models. It needs saved models, predictor metadata, point inputs and stacks. `--skip-hunting-importance` skips that diagnostic; `--resume-output` with the same `--run-id` can continue a compatible finalization.

The finalizer defaults to ten repeats; the example explicitly requests five to match the manuscript's repeat count. This hunting-only diagnostic is not a full-feature S6b exporter. The older [feature-importance runner](scripts/run_feature_importance.py) operates on best folds and training/background inputs. Read [feature-importance scope](../docs/REPRODUCIBILITY.md#feature-importance-coverage) before associating either output with the final supplement.

## Run folders

| Directory | Contents |
|---|---|
| `00_RUN/` | Logs, provenance, audits and replacement register |
| `01_INPUT/` | Source/resolved configs and publication source index, if supplied |
| `02_MODELS/` | Isolated model runs |
| `03_MANUSCRIPT/` | Main-document export artifacts |
| `04_SUPPLEMENT/` | Supplement export artifacts |
| `05_DIAGNOSTICS/` | Hunting, importance and novelty diagnostics |
| `06_MACHINE_READABLE/` | Combined result tables |
| `07_ARCHIVE/` | Publication bundle and exclusion index |

These are local generated artifacts, excluded from Git. A run can supply a publication source index; the code-only repository does not include `current_publication_index.csv` or the original manuscript files. See [input requirements](input/README.md).

## Current manuscript numbering

The [crosswalk](../docs/MANUSCRIPT_CROSSWALK.md#historical-export-names-versus-current-numbering) reconciles historical directory names with the reviewed PDFs. LOYO performance is current **Table 3**, metrics are **S6a**, full importance is **S6b**, response curves are **S7**, annual maps are **S8**, and novelty is **S9**. A directory containing `S7_feature_provenance` is an internal audit, not current S7.

Directory names stay unchanged for compatibility with saved runs. The artifact manifest now includes a separate `document_reference` field and updated human-readable labels.

## Novelty and provenance

The review novelty wrapper uses the default 99th-percentile Mahalanobis reference in the core novelty script. The manuscript describes a maximum reference. Check the actual saved settings using the [novelty note](../docs/REPRODUCIBILITY.md#novelty-threshold). Keep source/resolved YAMLs, model feature order, input checksums and environment records with every export.
