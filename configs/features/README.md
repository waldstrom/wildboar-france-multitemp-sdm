# Shared feature-selection rules

[Config index](../README.md) · [Predictor reference](../../docs/PREDICTORS.md)

[featureconfig.yaml](featureconfig.yaml) records candidate keep/drop rules and groups used by variable selection and temporal-stack processing (S4). All migrated model configs now reference `configs/features/featureconfig.yaml` relative to the repository root.

The file preserves its existing scientific rules. A `keep` rule is not a guarantee of final retention after correlation/VIF pruning. Preserve the rule file, resolved candidate set and each model's selected names/order when archiving results. Year-specific and harmonized names must be reconciled using actual stack/model metadata.
