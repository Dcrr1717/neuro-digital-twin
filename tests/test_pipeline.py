"""End-to-end test of the experiment pipeline on a seconds-scale configuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurotwin.__main__ import main
from neurotwin.config import config_from_dict
from neurotwin.pipeline import load_twin, render_report, run, select_model, tiny_config_dict


def test_select_model_prefers_gamma_then_rmse() -> None:
    a = {"gamma_mean": 0.9, "rmse_mv_median": 5.0}
    b = {"gamma_mean": 0.7, "rmse_mv_median": 1.0}
    assert select_model({"gru": a, "node": b}) == "gru"
    c = {"gamma_mean": 0.895, "rmse_mv_median": 1.0}
    assert select_model({"gru": a, "node": c}) == "node"


def test_select_model_enforces_speed_requirement() -> None:
    accurate_slow = {"gamma_mean": 0.95, "rmse_mv_median": 1.6}
    fast = {"gamma_mean": 0.93, "rmse_mv_median": 2.2}
    scores = {"gru": fast, "node": accurate_slow}
    assert select_model(scores) == "node"
    assert select_model(scores, {"gru": 300.0, "node": 20.0}, 100.0) == "gru"
    # If no candidate is fast enough, fall back to accuracy alone.
    assert select_model(scores, {"gru": 50.0, "node": 20.0}, 100.0) == "node"


def test_end_to_end_tiny(tmp_path: Path) -> None:
    cfg = config_from_dict(tiny_config_dict())
    m = run(cfg, tmp_path, log=lambda _: None)
    for f in ("metrics.json", "report.md"):
        assert (tmp_path / f).exists()
    for f in ("traces_true_vs_surrogate", "error_vs_parameter", "speedup", "uncertainty",
              "model_comparison", "fi_curves", "failure_modes"):
        assert (tmp_path / "figures" / f"{f}.png").stat().st_size > 1000
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["selection"]["winner"] in ("gru", "node")
    assert set(saved["selection"]["speedup_vs_scipy"]) == {"gru", "node"}
    assert set(saved["splits"]) == {"val", "test_id", "test_ood_param", "test_ood_stim"}
    assert saved["speed"]["scipy_lsoda"]["speedup_vs_scipy"] == 1.0
    assert saved["reference_agreement"]["max_abs_err_mv"] < 0.05
    assert "Speed" in render_report(m)

    twin = load_twin(tmp_path / "twin")
    assert len(twin.members) == cfg.ensemble.members
    theta = np.array([[120.0, 36.0, 0.3, 1.0]])
    mean, _std, _ = twin.predict(theta, np.zeros((1, cfg.sim.n_steps)))
    assert mean.shape == (1, cfg.sim.n_steps) and np.all(np.isfinite(mean))


def test_cli_runs_tiny_config(tmp_path: Path) -> None:
    import yaml

    cfg_path = tmp_path / "tiny.yaml"
    # parallel_jobs > 1 exercises the multi-process training path.
    cfg_path.write_text(yaml.safe_dump({**tiny_config_dict(), "parallel_jobs": 2}))
    out = tmp_path / "out"
    assert main(["run", "--config", str(cfg_path), "--out", str(out), "--no-figures"]) == 0
    assert (out / "metrics.json").exists() and not (out / "figures").exists()
    twin = load_twin(out / "twin")
    assert len(twin.members) == 2
