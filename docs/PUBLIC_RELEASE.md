# Publishing this code without publishing the data

The current repository has historically contained research data and derived geospatial files. Deleting those files in a later commit removes them from the current checkout, **not from Git history**. Therefore, do not make the existing repository public merely because the publication branch is data-free.

## Recommended release procedure

1. Review and merge the data-free publication changes into the private working branch if desired.
2. Create a history-free export of the approved commit:

   ```bash
   bash scripts/export_code_only.sh ../maxent-code-only
   ```

   On Windows PowerShell:

   ```powershell
   .\scripts\export_code_only.ps1 -Destination ..\maxent-code-only
   ```

3. Inspect the exported directory and run the included validator again.
4. Create a new, empty public repository and initialise the export as a new Git history:

   ```bash
   cd ../maxent-code-only
   git init -b main
   git add .
   git commit -m "Initial public code release"
   git remote add origin <NEW-EMPTY-REPOSITORY-URL>
   git push -u origin main
   ```

5. Select and add an open-source licence only after confirming institutional, collaborator and third-party obligations.
6. Add the article DOI and software release tag when available.

## Why a new repository is safer

A new root commit excludes historical data blobs, old generated outputs, deleted credentials and accidental local paths by construction. Rewriting the complete history of the private research repository is possible with tools such as `git filter-repo`, but it is disruptive, invalidates commit hashes and requires every clone and remote reference to be coordinated. A clean export avoids that risk.

## Final publication checks

Run:

```bash
python scripts/validate_repository.py
python -m compileall -q scripts review
```

Then inspect the complete tracked-file list:

```bash
git ls-files
```

The public repository should contain source code, YAML configuration, tests and Markdown documentation only. It should not contain occurrence records, raster/vector layers, model artefacts, caches, figures, office documents or publication working files.