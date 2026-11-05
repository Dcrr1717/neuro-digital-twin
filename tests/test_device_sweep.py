"""Device handling (CPU/GPU) and the patch-size sweep."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from neurotwin.__main__ import main
from neurotwin.config import config_from_dict
from neurotwin.models import GRUSurrogate, Normalizer, on_device, predict_mv, resolve_device
from neurotwin.pipeline import tiny_config_dict
from neurotwin.sweep import best_patch, render_sweep, run_patch_sweep

HAS_CUDA = torch.cuda.is_available()


def test_resolve_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type == ("cuda" if HAS_CUDA else "cpu")
    with pytest.raises(ValueError):
        resolve_device("meta")


@pytest.mark.skipif(HAS_CUDA, reason="needs a machine without CUDA")
def test_cuda_request_without_gpu_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match="no CUDA device"):
        resolve_device("cuda")


def test_config_validates_device() -> None:
    assert config_from_dict({"device": "cpu"}).device == "cpu"
    assert config_from_dict({"device": "cuda:1"}).device == "cuda:1"
    with pytest.raises(ValueError):
        config_from_dict({"device": "tpu"})


def test_on_device_returns_same_model_or_equal_copy() -> None:
    m = GRUSurrogate(hidden=8)
    assert on_device(m, "cpu") is m
    if HAS_CUDA:
        g = on_device(m, "cuda")
        assert g is not m and next(g.parameters()).is_cuda
        assert on_device(g, "cuda") is g


@pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
def test_gpu_prediction_matches_cpu() -> None:
    torch.manual_seed(0)
    norm = Normalizer([100, 28, 0.2, 0.85], [140, 44, 0.4, 1.15], 5.0,
                      [-60.0, 0.1, 0.5, 0.3], [15.0, 0.2, 0.2, 0.1])
    m = GRUSurrogate(hidden=16)
    theta = np.array([[120.0, 36.0, 0.3, 1.0]] * 3)
    cur = np.random.default_rng(0).normal(size=(3, 50))
    cpu = predict_mv(m, norm, theta, cur)
    gpu = predict_mv(on_device(m, "cuda"), norm, theta, cur)
    np.testing.assert_allclose(cpu, gpu, atol=1e-3)


def test_cli_overrides() -> None:
    from neurotwin.__main__ import apply_overrides

    overrides = ["gru.patch=10", "name=x", "eval.min_speedup=1.5"]
    raw = apply_overrides({"gru": {"hidden": 8}}, overrides)
    assert raw == {"gru": {"hidden": 8, "patch": 10}, "name": "x", "eval": {"min_speedup": 1.5}}
    with pytest.raises(SystemExit):
        apply_overrides({}, ["novalue"])


def test_best_patch_prefers_fastest_equally_accurate() -> None:
    def row(p: int, gamma: float, ms: float) -> dict[str, object]:
        return {"patch": p, "val": {"gamma_mean": gamma}, "ms_per_trace": {"cpu_b1": 9.0,
                                                                          "cpu_b1024": ms}}
    result = {"rows": [row(5, 0.948, 0.54), row(10, 0.945, 0.30), row(20, 0.90, 0.15)]}
    assert best_patch(result) == 10  # 20 is faster but clearly less accurate
    assert best_patch(result, tie=0.1) == 20


def test_sweep_cli_on_tiny_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny_config_dict()))
    out = tmp_path / "out"
    assert main(["sweep", "--config", str(cfg_path), "--out", str(out), "--device", "cpu",
                 "--patches", "5", "10"]) == 0
    result = json.loads((out / "patch_sweep.json").read_text())
    assert [r["patch"] for r in result["rows"]] == [5, 10]
    assert [r["recurrent_steps"] for r in result["rows"]] == [40, 20]
    for r in result["rows"]:
        assert set(r["ms_per_trace"]) >= {"cpu_b1", "cpu_b8"}
        assert 0.0 <= r["test_id"]["gamma_mean"] <= 1.0
    assert (out / "figures" / "patch_sweep.png").stat().st_size > 1000
    assert "| step (ms) |" in render_sweep(result)
    assert (out / "patch_sweep.md").exists()


def test_sweep_rejects_patch_that_does_not_divide_trace(tmp_path: Path) -> None:
    cfg = config_from_dict(tiny_config_dict())  # 200 samples per trace
    with pytest.raises(ValueError, match="do not divide"):
        run_patch_sweep(cfg, tmp_path, [5, 7], log=lambda _: None)
