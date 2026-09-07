# Aligning GBM and MaxEnt Outputs

This note summarises the main reasons why the gradient boosting (GBM) workflow
was diverging from the MaxEnt runs and documents the changes introduced in this
repository update. It also outlines a checklist to keep both engines comparable
when new experiments are executed.

## Why outputs looked different

* **Probability scale mismatch.** LightGBM returns probabilities that reflect
the raw training prevalence (after distance weights), whereas the MaxEnt
configuration exports cloglog-transformed scores. With a strong class
imbalance, the GBM raster therefore collapsed towards zero even though the rank
order was competitive.
* **Overfitting symptoms.** The default GBM parameters favoured deep trees with
low leaf regularisation. Training AUC/CBI were high, but validation/test
metrics degraded because the trees could memorise local artefacts.
* **Limited diagnostics.** The previous training log only printed the overall
AUC, making it hard to understand how the prediction distribution differed
between presences and background samples.

## What changed in this update

* **Platt calibration option.** `scripts/gbm_training.py` can now fit a logistic
calibration on the LightGBM raw scores. When enabled (the new default in
`config_gbm-compare.yaml`), the model learns an intercept and slope on the
validation hold-out and applies the correction whenever predictions are
requested. This aligns the mean suitability with MaxEnt's probability scale and
prevents the "mostly zero" rasters.【F:scripts/gbm_training.py†L32-L146】【F:scripts/gbm_training.py†L330-L386】
* **Richer diagnostics.** The training routine records probability summaries for
presences/backgrounds on the train and validation splits and logs split-specific
AUC/CBI values. These diagnostics make it immediately visible when calibration
or regularisation fails to separate classes.【F:scripts/gbm_training.py†L360-L407】
* **Stronger regularisation defaults.** The shared configuration now caps tree
depth, increases the minimum samples per leaf, and adds L1/L2 penalties. A
larger validation fraction supplies more data for early stopping and the new
calibration step.【F:config_gbm-compare.yaml†L221-L245】【F:config_gbm-compare.yaml†L461-L485】

## How to run comparable experiments

1. **Keep the calibration hold-out.** Retain `validation_fraction >= 0.2` so the
Platt scaling uses unseen data. If you change the split, adjust the
`gbm.calibration.dataset` flag accordingly.
2. **Compare score distributions.** After training, inspect the logged
probability percentiles for presences and background samples. Both engines
should now produce overlapping but not identical distributions; large divergences
flag misconfiguration.
3. **Track split metrics.** Use the split-specific AUC/CBI values to quantify
any residual overfitting. If the validation CBI is still markedly lower than
MaxEnt, tighten `num_leaves`, `lambda_l1`, or `lambda_l2` further.
4. **Synchronise thresholds.** Continue using the automatic F1 threshold (CBI
optimisation) from `scripts.evaluation`, which both engines now share.
5. **Archive calibration parameters.** The GBM model pickle stores the Platt
coefficients, so downstream scoring (e.g., in Java exports) keeps the corrected
scale without additional processing.【F:scripts/gbm_training.py†L68-L119】

Following this checklist keeps GBM and MaxEnt predictions on a comparable scale
and reduces the gap between their validation/test diagnostics.
