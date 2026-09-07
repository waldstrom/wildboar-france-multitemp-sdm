import importlib.machinery
import types
from pathlib import Path

import numpy as np
import optuna
import pytest
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "bias-strength-multi-optuna.py"
    loader = importlib.machinery.SourceFileLoader("bias_strength_multi_optuna", str(path))
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module


module = _load_module()


def _trial(bias: float, auc: float, cbi_p: float | None, cbi_s: float, *, state=TrialState.COMPLETE):
    user_attrs = {"test_auc": auc, "test_cbi_spearman": cbi_s}
    if cbi_p is not None:
        user_attrs["test_cbi_pearson"] = cbi_p
    else:
        user_attrs["test_cbi"] = auc / 10  # fallback target distinct from pearson

    return optuna.trial.create_trial(
        params={"bias_strength": bias},
        distributions={"bias_strength": FloatDistribution(0.0, 1.0)},
        value=auc,
        user_attrs=user_attrs,
        state=state,
    )


def test_resolve_config_paths_defaults_and_overrides():
    defaults = module._resolve_config_paths(None)
    assert defaults == list(module.DEFAULT_CONFIG_PATHS)

    custom = module._resolve_config_paths(["custom.yaml"])  # type: ignore[arg-type]
    assert custom == [Path("custom.yaml")]


def test_summary_tables_include_auc_and_cbi_statistics():
    trials = [
        _trial(0.0, 0.72, 0.15, 0.14),
        _trial(0.5, 0.81, 0.22, 0.2),
        _trial(0.5, 0.79, 0.18, 0.16),
        _trial(0.8, 0.75, 0.19, 0.17, state=TrialState.FAIL),
    ]

    metrics_df, summary_df = module._summaries_from_trials(trials)

    assert list(metrics_df.columns) == ["bias_strength", "auc", "cbi_pearson", "cbi_spearman"]
    assert metrics_df["bias_strength"].tolist() == [0.0, 0.5, 0.5]

    assert list(summary_df.columns) == [
        "bias_strength",
        "auc_mean",
        "auc_std",
        "cbi_pearson_mean",
        "cbi_pearson_std",
        "cbi_spearman_mean",
        "cbi_spearman_std",
        "n_trials",
    ]

    group = summary_df.loc[summary_df["bias_strength"] == 0.5].iloc[0]
    assert group["n_trials"] == 2
    assert group["auc_mean"] == pytest.approx(0.8)
    assert group["auc_std"] == pytest.approx(np.std([0.81, 0.79], ddof=1))
    assert group["cbi_pearson_mean"] == pytest.approx(0.2)
    assert group["cbi_pearson_std"] == pytest.approx(np.std([0.22, 0.18], ddof=1))
    assert group["cbi_spearman_mean"] == pytest.approx(0.18)


def test_summary_tables_fallback_to_single_cbi_value():
    trials = [_trial(0.2, 0.7, None, 0.12)]

    metrics_df, summary_df = module._summaries_from_trials(trials)

    assert metrics_df.loc[0, "cbi_pearson"] == pytest.approx(0.07)
    assert summary_df.loc[0, "cbi_pearson_mean"] == pytest.approx(0.07)
