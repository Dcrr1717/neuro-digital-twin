"""Patch-size sweep: how much speed does a coarser GRU time step buy, at what accuracy?

The GRU twin reads the stimulus and emits the voltage in *patches* of
``patch`` samples per recurrent step (0.5 ms at the default 5 x 0.1 ms).
Longer patches mean fewer sequential steps per trace and therefore faster
inference, at the price of a harder read-out (each step must render a
longer stretch of the trace, possibly a whole action potential).

``run_patch_sweep`` trains one GRU per patch size on the same data, seed and
number of epochs as the main experiment and reports accuracy on every held-out
split together with the inference cost on the CPU and, when PyTorch sees a
CUDA device, on the GPU.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import torch

from . import plots
from .config import Config
from .data import build_datasets
from .evaluate import aggregate, tile_inputs, time_call, trace_metrics
from .hh import HHParams, simulate_scipy
from .models import Normalizer, device_label, on_device, predict_mv, resolve_device
from .pipeline import train_many

SPLITS = ("val", "test_id", "test_ood_param", "test_ood_stim")


def run_patch_sweep(
    cfg: Config,
    out: Path,
    patches: Sequence[int],
    figdir: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train one GRU per patch size and measure accuracy and inference speed.

    Data and trained models are cached under ``out`` exactly as in
    :func:`neurotwin.pipeline.run`, so running the sweep in the output
    directory of a finished run reuses its datasets and its patch-5 model.
    """
    t_start = time.perf_counter()
    figdir = figdir or out / "figures"
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
    n_steps = cfg.sim.n_steps
    bad = [p for p in patches if p < 1 or n_steps % p]
    if bad:
        raise ValueError(f"patch sizes {bad} do not divide the {n_steps} samples per trace")
    device = resolve_device(cfg.device)
    dt = cfg.sim.dt_ms
    log(f"patch sweep {list(patches)} ({cfg.name}; training on {device_label(device)})")

    data = build_datasets(cfg, out / "data")
    lo, hi = cfg.params.bounds("train")
    norm = Normalizer.fit(lo, hi, data["train"].current, data["train"].states)
    cfgs = [replace(cfg, gru=replace(cfg.gru, patch=int(p))) for p in patches]
    seed = cfg.seed * 1000 + 1
    trained = train_many([("gru", seed)] * len(cfgs), data, norm, cfg, log,
                         cache=out / "models", job_cfgs=cfgs)

    test = data["test_id"]
    n_ref = min(cfg.eval.speed_n_scipy, len(test))
    base = HHParams(temperature=cfg.sim.temperature)
    t0 = time.perf_counter()
    for k in range(n_ref):
        simulate_scipy(test.theta[k], test.current[k], dt, base, cfg.sim.rtol, cfg.sim.atol)
    scipy_s = (time.perf_counter() - t0) / n_ref
    log(f"  scipy LSODA: {scipy_s * 1e3:.1f} ms/trace")

    big = max(cfg.eval.speed_batch_sizes)
    has_gpu = torch.cuda.is_available()
    rows: list[dict[str, Any]] = []
    for p, res in zip(patches, trained, strict=True):
        row: dict[str, Any] = {
            "patch": int(p), "step_ms": round(p * dt, 6), "recurrent_steps": n_steps // p,
            "train_seconds": round(res.seconds, 1), "best_epoch": res.best_epoch,
        }
        for split in SPLITS:
            ds = data[split]
            pred = predict_mv(res.model, norm, ds.theta, ds.current)
            agg = aggregate(trace_metrics(ds.voltage, pred, dt, cfg.eval.coincidence_window_ms))
            row[split] = {k: round(agg[k], 4) for k in
                          ("gamma_mean", "spike_count_exact", "rmse_mv_median",
                           "sub_rmse_mv_median")}
        speed: dict[str, float] = {}
        cpu_model = on_device(res.model, "cpu")
        for bs in (1, big):
            th, cur = tile_inputs(test, bs)
            t = time_call(partial(predict_mv, cpu_model, norm, th, cur, bs))
            speed[f"cpu_b{bs}"] = t / bs
        if has_gpu:
            gpu_model = on_device(res.model, "cuda")
            for bs in cfg.eval.gpu_batch_sizes:
                th, cur = tile_inputs(test, bs)
                t = time_call(partial(predict_mv, gpu_model, norm, th, cur, bs))
                speed[f"cuda_b{bs}"] = t / bs
        row["ms_per_trace"] = {k: round(v * 1e3, 5) for k, v in speed.items()}
        row["speedup_vs_scipy"] = {k: round(scipy_s / v, 1) for k, v in speed.items()}
        rows.append(row)
        log(f"  patch {p} ({p * dt:g} ms/step): val Gamma={row['val']['gamma_mean']:.3f} "
            f"test Gamma={row['test_id']['gamma_mean']:.3f} "
            f"RMSE={row['test_id']['rmse_mv_median']:.2f} mV | "
            + ", ".join(f"{k} {v:.4g} ms" for k, v in row["ms_per_trace"].items()))

    result: dict[str, Any] = {
        "config": cfg.name,
        "device_trained": device_label(device),
        "gpu": device_label(torch.device("cuda")) if has_gpu else None,
        "cpu_threads": torch.get_num_threads(),
        "scipy_ms_per_trace": round(scipy_s * 1e3, 3),
        "epochs": cfg.gru.epochs,
        "rows": rows,
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "patch_sweep.json").write_text(json.dumps(result, indent=2))
    (out / "patch_sweep.md").write_text(render_sweep(result))
    plots.plot_patch_sweep(figdir / "patch_sweep.png", result)
    log(f"done in {result['seconds']:.0f} s -> {out / 'patch_sweep.json'}")
    return result


def render_sweep(result: dict[str, Any]) -> str:
    """Markdown table of a sweep result."""
    keys = list(result["rows"][0]["ms_per_trace"])
    head = ("| step (ms) | GRU steps / trace | val Γ | test Γ | test exact count | "
            "test median RMSE (mV) | extrap. params Γ | novel stimuli Γ | "
            + " | ".join(f"{k} ms/trace (speed-up)" for k in keys) + " |")
    lines = [f"# Patch sweep ({result['config']} config)", "",
             f"CPU threads: {result['cpu_threads']}; GPU: {result['gpu'] or 'none available'}; "
             f"scipy LSODA: {result['scipy_ms_per_trace']} ms/trace; "
             f"{result['epochs']} epochs per model.", "", head,
             "|" + "---|" * (8 + len(keys))]
    for r in result["rows"]:
        cells = [f"{r['step_ms']:g}", str(r["recurrent_steps"]),
                 f"{r['val']['gamma_mean']:.3f}", f"{r['test_id']['gamma_mean']:.3f}",
                 f"{r['test_id']['spike_count_exact']:.2f}",
                 f"{r['test_id']['rmse_mv_median']:.2f}",
                 f"{r['test_ood_param']['gamma_mean']:.3f}",
                 f"{r['test_ood_stim']['gamma_mean']:.3f}"]
        cells += [f"{r['ms_per_trace'][k]:.4g} ({r['speedup_vs_scipy'][k]:,.0f}×)" for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def best_patch(result: dict[str, Any], tie: float = 0.01, key: str | None = None) -> int:
    """Fastest patch whose validation Gamma is within ``tie`` of the best."""
    rows = result["rows"]
    best_gamma = max(r["val"]["gamma_mean"] for r in rows)
    ok = [r for r in rows if best_gamma - r["val"]["gamma_mean"] < tie]
    key = key or max(rows[0]["ms_per_trace"], key=lambda k: int(k.split("_b")[1]))
    return int(min(ok, key=lambda r: r["ms_per_trace"][key])["patch"])

