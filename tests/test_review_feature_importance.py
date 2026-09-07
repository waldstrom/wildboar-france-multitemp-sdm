import pytest

from review.scripts.run_feature_importance import _predictors_for_model


class _Booster:
    def __init__(self, names):
        self._names = names

    def feature_name(self):
        return self._names


class _GBM:
    def __init__(self, names):
        self.booster = _Booster(names)


def test_predictors_for_model_uses_fitted_order_and_subset():
    model = _GBM(["third", "first"])

    assert _predictors_for_model(model, ["first", "second", "third"]) == [
        "third",
        "first",
    ]


def test_predictors_for_model_rejects_missing_fitted_predictor():
    with pytest.raises(RuntimeError, match="missing"):
        _predictors_for_model(_GBM(["present", "missing"]), ["present"])


def test_predictors_for_model_falls_back_for_models_without_names():
    selected = ["first", "second"]

    assert _predictors_for_model(object(), selected) is selected
