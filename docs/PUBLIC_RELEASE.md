# Public code releases

[Repository home](../README.md) · [Licence](LICENSING.md) · [Reproduction status](REPRODUCIBILITY.md)

This repository is the public code-only companion. Its working tree contains code, configurations, documentation and tests; input data and generated research outputs belong in separate authorized storage.

## Before a release

```bash
python scripts/validate_repository.py
python scripts/validate_documentation.py
python -m compileall -q scripts review review-corrections.py review-finalize-existing-run.py
```

Run relevant synthetic tests for code changes. Check the diff for missing configuration references, accidental outputs and source-specific licence notices. Record the exact commit used for a release. Repository checks do not establish numerical reproduction of the manuscript.

## Exporting the approved commit

The existing export helpers produce an independent code tree:

```bash
bash scripts/export_code_only.sh ../wildboar-france-code-only
```

Windows PowerShell:

```powershell
.\scripts\export_code_only.ps1 -Destination ..\wildboar-france-code-only
```

Use a new destination directory; the helpers refuse to overwrite an existing one. They use the committed tree; uncommitted changes are not an approved release snapshot. The [ZIP workflow](../.github/workflows/build-data-free-zip.yml) also produces a code-only archive for main-branch pull requests or a manual workflow dispatch. It validates the archive contents before upload.

Retain [LICENSE](../LICENSE), [CITATION.cff](../CITATION.cff), [codemeta.json](../codemeta.json), data acquisition guides and the config catalog in exports. The configuration moves do not remove the legacy experiment records.

## Archival metadata

When a real versioned release is made, record its tag/date and archive DOI in software metadata. Keep the preferred study DOI separate. Do not use the preprint DOI as a software identifier or imply the reviewed manuscript has a final journal DOI before that is verified.

Store permitted research outputs in an appropriate separate archive, with producing commit/configuration, environment and source-data terms. A data-free software archive does not by itself reproduce maps and tables.

## Exporting from the private research repository

If creating another public snapshot from a private working repository, use a fresh history-free export. Deleting data in a later commit does not remove earlier blobs from Git history. This warning concerns exporting private research history; it does not assert that the current public repository contains restricted data in its history.
