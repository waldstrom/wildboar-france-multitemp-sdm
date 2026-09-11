# Manuscript and supplement crosswalk

[Documentation home](README.md) · [Methods](METHODS.md) · [Outputs](OUTPUTS.md)

References below use the reviewed PDFs identified on the [documentation home](README.md#documentation-basis). They were checked against the supplied text and public code snapshot; a listed module may implement only part of a publication stage. No manuscript PDF or result data are distributed here.

## Main manuscript

| Item / location | Subject | Code, configuration and provenance |
|---|---|---|
| Fig. 1; Methods lines 134-145 | Integrated study workflow | [Methods](METHODS.md); [script index](../scripts/README.md) |
| Table 1; Fig. 2; lines 149-186 | Occurrences, seasons, thinning and retention | [thinning.py](../scripts/thinning.py), [utils.py](../scripts/utils.py); `presence_data` settings; [data guide](../data/README.md). Exact counts require prepared occurrence records |
| Fig. 3; lines 194-231 | Source-specific and combined bias maps | [bias_grid.py](../scripts/bias_grid.py), [background_sampling.py](../scripts/background_sampling.py); [bias inputs](../data/bias-maps/README.md). Upstream traffic/accessibility processing is not fully bundled |
| Fig. 4; lines 289-320 | Six-domain environmental workflow | [preprocessing.py](../scripts/preprocessing.py), [variable_selection.py](../scripts/variable_selection.py), [feature config](../configs/features/featureconfig.yaml); [predictor reference](PREDICTORS.md) |
| Table 2; Fig. 5; lines 263-285, 385-420 | ViT land-cover training, metrics and maps | [Land-cover input guide](../data/land-cover/README.md). Training imagery, labels, checkpoints, confusion matrix and the full ViT training workflow are outside this snapshot |
| Methods lines 335-358 | Temporal representations and nested evaluation | [multitemporal.py](../scripts/multitemporal.py), [monotemporal.py](../scripts/monotemporal.py), [tuning guide](../configs/tuning/README.md). Read [tuning limitations](REPRODUCIBILITY.md#tuning-and-outer-evaluation) |
| Table 3; lines 421-455 | Held-out-year SDM performance | [LOYO meta runner](../scripts/meta_run_leave_one_year.py), [evaluation.py](../scripts/evaluation.py), [review finalizer](../review-finalize-existing-run.py); [reviewed presets](../configs/reviewed/README.md) |
| Fig. 6; lines 460-479 | 2019 relative suitability comparison | Temporal runners' prediction export and [visualization.py](../scripts/visualization.py); saved test-2019 models and rasters |
| Fig. 7; lines 481-489 | WVC+GBIF versus GBIF-only difference | Matched saved 2019 winter predictions; preserve mask, transform and subtraction direction. A dedicated complete publication-layout recipe is not established by this snapshot |
| Tables 4-5; lines 369-373, 494-549 | Mean winter/summer feature importance | [variable_importance.py](../scripts/variable_importance.py), [aggregation helper](../scripts/analyze_feature_importance.py), [hunting finalizer](../review-finalize-existing-run.py); [coverage caveats](OUTPUTS.md#feature-importance) |
| Fig. 8; lines 375-383, 550-570 | NT1/NT2 novelty | [mess_evaluation.py](../scripts/mess_evaluation.py), [review novelty wrapper](../review/scripts/run_novelty.py); [threshold difference](REPRODUCIBILITY.md#novelty-threshold) |
| Lines 844-854 | Data and code availability | [data guide](../data/README.md), [licensing](LICENSING.md), [release policy](PUBLIC_RELEASE.md) |

## Supplement

Page numbers are the printed supplement page numbers.

| Section | Starts | Contents | Repository reference |
|---|---:|---|---|
| S1 | 5 | Occurrence thinning algorithm | [thinning.py](../scripts/thinning.py); `presence_data.thinning`; [occurrence contract](../data/README.md#occurrence-records-allpointscsv) |
| S2 | 6 | Bias correction and source surfaces | [bias_grid.py](../scripts/bias_grid.py), [background_sampling.py](../scripts/background_sampling.py); [bias data guide](../data/bias-maps/README.md) |
| S3a | 10 | ViT reference-label composition | [land-cover data guide](../data/land-cover/README.md); source labels outside this snapshot |
| S3b | 10 onward | ViT model/training/inference configuration | External classifier workflow and assets; [scope note](REPRODUCIBILITY.md#upstream-data-and-vision-transformer) |
| S3c | Within S3 | Classification evaluation and manual interpretation | External classification evaluation; not SDM LOYO metrics |
| S4 | 20 | Environmental SDM features | [Predictors](PREDICTORS.md), [feature config](../configs/features/featureconfig.yaml), `predictors.layers` |
| S5 | 36 | Example comparative config and search ranges | [reviewed presets](../configs/reviewed/README.md); historical [all-source aggregate](../configs/experiments/config_gbm_loyo_compare.yaml); [tuning guide](../configs/tuning/README.md) |
| S6a | 47 | LOYO performance by year | `temporal_cv_metrics.csv`; [LOYO runner](../scripts/meta_run_leave_one_year.py), [review finalizer](../review-finalize-existing-run.py) |
| S6b | 52 | Full LOYO permutation-importance rankings | [importance reference](OUTPUTS.md#feature-importance). Requires all expected held-out years and explicit missing-feature treatment |
| S7 | 60 | Variable response curves | [visualization.py](../scripts/visualization.py) and temporal model outputs |
| S8 | 64 | Full annual SDM output maps | Seasonal prediction rasters and saved fold models; [output reference](OUTPUTS.md) |
| S9 | 72 | Extrapolation novelty | [mess_evaluation.py](../scripts/mess_evaluation.py); saved reference statistics and threshold metadata |

## Historical export names versus current numbering

The review package predates final manuscript numbering. Keep existing paths stable for saved-run compatibility; interpret them using this map.

| Historical export name or label | Current reviewed-document meaning |
|---|---|
| `01_Table_2_LOYO_model_performance` | Main **Table 3**, SDM performance. Current Table 2 is ViT classification |
| `Supplement_S6_foldwise_performance_and_CI` | **S6a**, LOYO metrics |
| `Supplement_S7_feature_provenance_and_hunting_audit` | Internal candidate/retention audit. It is not current S7 |
| `S7_hunting_bag_all_LOYO_folds...` | Hunting diagnostic supporting interpretation of **S6b**; not the response-curve section |
| `Supplement_response_curves_and_full_importance` | Response curves belong to **S7**; full importance tables belong to **S6b** |
| `Supplement_full_corrected_winter_maps` | Winter subset of **S8** |
| `Supplement_MESS_NT1_NT2_novelty` | **S9**, also supporting Fig. 8 |

The [artifact manifest](../review/config/artifact-manifest.yaml) retains stable directory names and gives the current document references separately. Do not mechanically insert an old export label into the revised manuscript.
