# Review-workflow inputs — intentionally empty

No manuscript extracts, publication indexes, experiment tables, model artefacts or other generated inputs are tracked here.

The review utilities may expect locally generated files including:

- `current_publication_index.csv`, produced by `python -m review.scripts.index_publication_sources`
- a list of current source files or run directories
- model predictions, feature-importance tables and run metadata referenced by `review/config/`

Create these from authorised local runs and keep them untracked. Consult `review/README.md` and the active files in `review/config/` for the exact command and expected columns. Never place reviewer correspondence, unpublished manuscript files or restricted occurrence data in Git.