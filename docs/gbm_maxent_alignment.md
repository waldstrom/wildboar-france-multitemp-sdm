# Interpreting MaxEnt and GBM outputs

[Documentation home](README.md) · [Methods](METHODS.md) · [Reproduction status](REPRODUCIBILITY.md#engine-scales)

Both engines are fitted with presence-background data. Their bounded predictions are not calibrated estimates of species occurrence probability.

| Engine | Implemented prediction | Interpretation |
|---|---|---|
| MaxEnt (`elapid`) | Configured output transformation, `cloglog` in study presets | Relative habitat suitability against the sampled background |
| LightGBM, uncalibrated | Booster output from `GBMModel.predict` | Score tied to the fitted presence-background classification problem |
| LightGBM, Platt calibration | Logistic intercept/slope adjustment of raw scores | Calibration to the chosen presence-background split; not ecological prevalence calibration |

See [maxent_training.py](../scripts/maxent_training.py) and [gbm_training.py](../scripts/gbm_training.py). The latter supports Platt calibration; unsupported calibration types raise an error. Do not infer isotonic support from S5's example search space.

The temporal export helpers share filenames with a `cloglog` suffix. Inspect the engine, model wrapper and resolved output settings rather than inferring scale from the filename. Platt calibration does not mathematically transform LightGBM into MaxEnt or guarantee identical score distributions.

For comparisons, retain the same occurrence/background design, outer folds, grid and mask; report each engine's score definition. Rank metrics and distribution diagnostics can support comparison, but matching ranges or means does not establish equivalent ecological calibration. Classification thresholds do not determine CBI in the way an F1 threshold determines binary classification metrics.
