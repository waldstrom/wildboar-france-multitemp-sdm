from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from review.scripts.artifacts import (
    _boxplot_with_tick_labels,
    _sort_by_available_columns,
    plot_performance,
    summarise_test_metrics,
)
from review.scripts.configuration import (
    discover_disabled_winter_monotemporal_configs,
    resolve_scenario,
    select_aggregate_scenario,
)


class ConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        aggregate = {
            "scenarios": [
                {
                    "key": "winter_mono",
                    "label": "Winter monotemporal",
                    "script": "monotemporal",
                    "config": {
                        "monotemporal": True,
                        "multitemporal": {
                            "seasons": ["Winter"],
                            "years": [2018, 2019],
                            "include_hunting": False,
                        },
                        "predictors": {
                            "layers": {
                                "hunting_bag": {
                                    "dir": "data",
                                    "pattern": "hunting_{year}.tif",
                                }
                            }
                        },
                    },
                },
                {
                    "key": "winter_multi",
                    "label": "Winter multitemporal",
                    "script": "multitemporal",
                    "config": {
                        "multitemporal": {
                            "seasons": ["Winter"],
                            "years": [2018, 2019],
                            "include_hunting": True,
                        }
                    },
                },
            ]
        }
        (self.root / "aggregate.yaml").write_text(
            yaml.safe_dump(aggregate, sort_keys=False), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_semantic_scenario_selection(self) -> None:
        cfg, item = select_aggregate_scenario(
            self.root / "aggregate.yaml",
            {"runner": "monotemporal", "season": "Winter"},
        )
        self.assertEqual(item["key"], "winter_mono")
        self.assertFalse(cfg["multitemporal"]["include_hunting"])

    def test_resolved_config_activates_hunting(self) -> None:
        spec = {
            "id": "winter_monotemporal_allpoints",
            "order": 1,
            "role": "corrected",
            "dataset": "allpoints",
            "temporal": "monotemporal",
            "runner": "monotemporal",
            "season": "Winter",
            "expect_hunting": True,
            "source": {
                "aggregate_config": "aggregate.yaml",
                "selector": {"runner": "monotemporal", "season": "Winter"},
            },
            "patch": {"multitemporal": {"include_hunting": True}},
        }
        record = resolve_scenario(
            self.root,
            spec,
            self.root / "review_scenario",
            run_id="TEST",
            default_engines=["elapid", "gbm"],
            default_seed=42,
        )
        resolved = yaml.safe_load(Path(record.resolved_config).read_text())
        self.assertTrue(resolved["multitemporal"]["include_hunting"])
        self.assertEqual(resolved["model"]["engines"], ["elapid", "gbm"])
        self.assertFalse(resolved["experiment"]["append_timestamp"])
        self.assertTrue(resolved["experiment"]["review_expect_hunting"])

    def test_repository_discovery_finds_disabled_setting(self) -> None:
        findings = discover_disabled_winter_monotemporal_configs(
            self.root, ["*.yaml"]
        )
        disabled = [row for row in findings if row.get("status") == "disabled"]
        self.assertEqual(len(disabled), 1)
        self.assertEqual(disabled[0]["scenario"], "winter_mono")


class MetricTests(unittest.TestCase):
    def test_summary_recomputes_ci(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset": ["allpoints"] * 3,
                "temporal": ["monotemporal"] * 3,
                "engine": ["elapid"] * 3,
                "season_review": ["Winter"] * 3,
                "split": ["test"] * 3,
                "test_year": [2018, 2019, 2020],
                "cbi_spearman": [0.7, 0.8, 0.9],
                "auc": [0.75, 0.80, 0.85],
            }
        )
        summary = summarise_test_metrics(frame)
        self.assertEqual(int(summary.loc[0, "n_folds"]), 3)
        self.assertAlmostEqual(float(summary.loc[0, "cbi_spearman_mean"]), 0.8)
        self.assertLess(
            float(summary.loc[0, "cbi_spearman_ci95_low"]),
            float(summary.loc[0, "cbi_spearman_mean"]),
        )
        self.assertGreater(
            float(summary.loc[0, "cbi_spearman_ci95_high"]),
            float(summary.loc[0, "cbi_spearman_mean"]),
        )

    def test_empty_metric_table_can_be_prepared_for_export(self) -> None:
        frame = pd.DataFrame()
        result = _sort_by_available_columns(
            frame, ["dataset", "engine", "temporal", "test_year", "split"]
        )
        self.assertTrue(result.empty)
        self.assertIsNot(result, frame)

    def test_metric_table_sorts_only_by_columns_that_exist(self) -> None:
        frame = pd.DataFrame({"dataset": ["gbif", "allpoints"], "value": [1, 2]})
        result = _sort_by_available_columns(frame, ["dataset", "engine"])
        self.assertEqual(result["dataset"].tolist(), ["allpoints", "gbif"])


class ArtifactPlotTests(unittest.TestCase):
    def test_boxplot_uses_current_matplotlib_tick_label_keyword(self) -> None:
        calls = []

        class CurrentAxes:
            def boxplot(self, groups, *, tick_labels=None, showmeans=False):
                calls.append((groups, tick_labels, showmeans))

        groups = [pd.Series([0.5, 0.7]).to_numpy()]
        _boxplot_with_tick_labels(CurrentAxes(), groups, ["GBIF"], showmeans=True)
        self.assertEqual(calls[0][1], ["GBIF"])
        self.assertTrue(calls[0][2])

    def test_boxplot_remains_compatible_with_legacy_matplotlib(self) -> None:
        calls = []

        class LegacyAxes:
            def boxplot(self, groups, *, labels=None, showmeans=False):
                calls.append((groups, labels, showmeans))

        groups = [pd.Series([0.5, 0.7]).to_numpy()]
        _boxplot_with_tick_labels(LegacyAxes(), groups, ["GBIF"], showmeans=True)
        self.assertEqual(calls[0][1], ["GBIF"])
        self.assertTrue(calls[0][2])

    def test_performance_plot_exports_png_and_pdf(self) -> None:
        metrics = pd.DataFrame(
            {
                "dataset": ["gbif"] * 3,
                "temporal": ["monotemporal"] * 3,
                "engine": ["elapid"] * 3,
                "split": ["test"] * 3,
                "cbi_spearman": [0.5, 0.6, 0.7],
                "auc": [0.7, 0.8, 0.9],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            outputs = plot_performance(metrics, Path(tmp), dpi=72)
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(path.is_file() and path.stat().st_size for path in outputs))


if __name__ == "__main__":
    unittest.main()
