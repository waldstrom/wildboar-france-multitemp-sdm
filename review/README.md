# Winter-monotemporal hunting correction

This package is designed to be copied directly into the root of `maxent-repo`.
Its only top-level executable is `review-corrections.py`; implementation files,
configuration, tests and documentation live below `review/`.

## What it corrects

The published winter comparison used `include_hunting: false` for the
winter-monotemporal configuration and `include_hunting: true` for the matched
winter-multitemporal configuration. The correction runner creates immutable
scenario copies with hunting activated and rebuilds all affected stacks and
models.

A predictor can be **activated as a candidate** and still be removed in an
individual LOYO fold by the existing correlation/VIF selection. The generated
S7 audit therefore reports separately:

- whether every annual hunting raster exists;
- whether a hunting-named variable entered the fresh master stack;
- whether the materialized fold stack contains it;
- whether it was retained in `selected_predictors.txt`.

This prevents “activated” from being confused with “forced into every model.”

## Default model set

The default command runs four matched winter scenarios, all with both `elapid`
and `gbm`:

1. winter monotemporal, GBIF + WVC — **corrected**;
2. winter multitemporal, GBIF + WVC — regenerated reference;
3. winter monotemporal, GBIF-only — **corrected**;
4. winter multitemporal, GBIF-only — regenerated reference.

Summer is not rerun because the reviewer-identified setting is confined to the
winter-monotemporal comparison. Two optional no-hunting controls reproduce the
old setting when `--include-controls` is supplied.

## First commands

```bash
python review-corrections.py self-test
python review-corrections.py plan
python review-corrections.py run --dry-run
python review-corrections.py run
```

The dry run resolves all embedded aggregate scenarios, writes immutable corrected
YAML files, inventories repository-wide disabled hunting settings, and checks the
package without executing models.

A failed run is resumable:

```bash
python review-corrections.py run --resume review/output/<RUN_ID>
```

Completed scenarios and diagnostics are skipped. Partial experiment directories
are archived before a fresh restart, so a stale Zarr stack is never silently
accepted.

Useful controlled variants:

```bash
# Run only the two corrected monotemporal models. Comparison figures requiring
# regenerated multitemporal references may be incomplete.
python review-corrections.py run --corrected-only

# Quantify the exact old-vs-corrected difference.
python review-corrections.py run --include-controls

# Re-export after changing only publication formatting.
python review-corrections.py export --run-dir review/output/<RUN_ID>

# Validate an existing run.
python review-corrections.py validate --run-dir review/output/<RUN_ID>
```

## Timestamped output tree

```text
review/output/<YYYYMMDD_HHMMSS_ZONE_hunting-correction>/
├── 00_RUN/              logs, provenance, audits, validation, replacement register
├── 01_INPUT/            source and resolved configurations, source-publication index
├── 02_MODELS/           isolated model runs in deterministic scenario order
├── 03_MANUSCRIPT/       replacement artifacts in manuscript order
├── 04_SUPPLEMENT/       S5, S6, S7 and further supplement exports in order
├── 05_DIAGNOSTICS/      hunting audit, feature importance, MESS/NT1/NT2
├── 06_MACHINE_READABLE/ combined corrected tables
└── 07_ARCHIVE/          compact publication bundle and large-file exclusion index
```

`review/output/LATEST.txt` points to the latest successful run.

## Publication artifacts that must be replaced

The package treats the following as affected and generates each programmatically:

1. the central LOYO performance table, including corrected winter-monotemporal
   rows and recomputed confidence intervals;
2. every performance figure containing winter-monotemporal CBI/AUC values;
3. the 2019 winter composite map wherever a monotemporal panel appears;
4. winter feature-importance outputs;
5. numerical values quoted in Results or Discussion;
6. Supplement S5 configuration excerpts;
7. Supplement S6 fold-wise metrics and confidence intervals;
8. Supplement S7 predictor provenance / selected-feature records;
9. winter response curves and full feature-importance files;
10. winter monotemporal MESS, limiting-variable, NT1/NT2 and comparison outputs;
11. all winter-monotemporal held-out-year prediction maps.

The included `review/input/current_publication_index.csv` is extracted from the
current manuscript DOCX and supplement PDF supplied with the review package. It
provides a searchable source index; the semantic export folders remain stable if
typesetting later changes figure numbering.

## Safety and reproducibility

- Original repository YAML files are never edited.
- Every run gets an isolated stack and experiment directory.
- Corrected scenarios always execute with `rebuild=True`.
- Both config snapshots and unified diffs are retained.
- Git commit, dirty status, Python environment and file hashes are recorded.
- A failed hunting-source or master-stack audit stops publication export by default.
- The compact ZIP excludes individual files larger than the configured threshold;
  the complete files remain in the timestamped run directory and are indexed.
