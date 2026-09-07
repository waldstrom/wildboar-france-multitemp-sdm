import importlib.machinery
import pathlib
import types


def _load_build_trial_params():
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "maha_thin_utils.py"
    loader = importlib.machinery.SourceFileLoader("maha_thin_utils", str(path))
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module._build_trial_params


def test_trial_param_order():
    _build_trial_params = _load_build_trial_params()
    sources = ["GBIF", "RRN", "SNCF"]
    years = list(range(2017, 2024))
    test_year = 2023
    variants = ["global"] + [f"{s}_{y}" for y in years if y != test_year for s in sources]
    params = _build_trial_params([0.4, 0.8, 1.0], variants)
    assert len(params) == 39
    assert params[0] == {"thin_pct": 0.4, "thin_variant_idx": 1}
    assert params[17] == {"thin_pct": 0.4, "thin_variant_idx": 18}
    assert params[18] == {"thin_pct": 0.8, "thin_variant_idx": 1}
    assert params[35] == {"thin_pct": 0.8, "thin_variant_idx": 18}
    assert params[36] == {"thin_pct": 1.0, "thin_variant_idx": 0}
    assert params[37] == {"thin_pct": 0.8, "thin_variant_idx": 0}
    assert params[38] == {"thin_pct": 0.4, "thin_variant_idx": 0}
