from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from review.scripts import run_novelty


@pytest.mark.parametrize(
    ("year_args", "expected_years"),
    [([], None), (["--year", "2019"], [2019])],
)
def test_main_passes_years_to_mess_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    year_args: list[str],
    expected_years: list[int] | None,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_mess(run_dir: Path, **kwargs: object) -> Path:
        calls.append({"run_dir": run_dir, **kwargs})
        return tmp_path / f"result-{len(calls)}"

    monkeypatch.setattr("scripts.mess_evaluation.run_mess", fake_run_mess)
    monkeypatch.setattr(
        "scripts.mess_evaluation.compare_runs",
        lambda *args, **kwargs: tmp_path / "comparison",
    )
    output_root = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_novelty",
            "--repo-root",
            str(tmp_path),
            "--mono-run",
            str(tmp_path / "mono"),
            "--multi-run",
            str(tmp_path / "multi"),
            "--output-root",
            str(output_root),
            "--dataset",
            "allpoints",
            *year_args,
        ],
    )

    original_cwd = Path.cwd()
    try:
        assert run_novelty.main() == 0
    finally:
        os.chdir(original_cwd)

    assert len(calls) == 2
    assert all(call["years"] == expected_years for call in calls)
    assert (output_root / "COMPLETE").is_file()
