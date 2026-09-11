# Changelog

## Unreleased

### Documentation and repository organization

- Add a documentation home, methods guide, quick start, predictor/output references and manuscript/supplement crosswalk based on the September 2026 reviewed PDFs.
- Move 64 YAML files into purpose-specific directories, preserving parsed scientific settings. Record every path and consumer in a 69-config catalog and migration guide.
- Repair invalid indentation in two summer tuning presets.
- Add two reviewed LOYO starting presets with corrected winter hunting inclusion, five configured permutation repeats and distinct experiment names; make them the LOYO meta runners' defaults.
- Document remaining differences in tuning objectives, feature-importance evaluation scope, novelty normalization, GBM score interpretation and upstream classifier coverage.
- Reconcile historical review export labels with current Table 3 and S6a/S6b/S7-S9 while preserving saved-run directory names.
- Add the existing README's intended BSD 3-Clause licence, CFF citation metadata and CodeMeta metadata without inventing a software release or DOI.
- Add config/catalog/link checks to CI, correct the ZIP workflow's target to `main`, and restore the required review output guide excluded by a nested ignore rule.
- Make code exports refuse existing destinations; update export naming and guidance for this public repository.

This update does not refit models, rerun the scientific analyses or alter fitting algorithms. Changed reviewed-preset values are explicit; historical settings remain available.
