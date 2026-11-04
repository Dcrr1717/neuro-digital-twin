"""Accuracy, uncertainty and speed evaluation of the digital twin."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import spearmanr

from .config import Config
from .data import Dataset
from .features import coincidence_factor, spike_mask, spike_times, summary_table
from .hh import HHParams, simulate_batch, simulate_scipy
from .models import Ensemble, Normalizer, Surrogate, on_device, predict_mv

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def trace_metrics(
    v_true: FloatArray, v_pred: FloatArray, dt: float, delta: float
) -> dict[str, FloatArray]:
    """Per-trace accuracy metrics (arrays of length ``B``).

    * ``rmse_mv``      -- root-mean-square voltage error over the whole trace.
    * ``sub_rmse_mv``  -- RMSE restricted to samples away from any spike
      (true or predicted): the subthreshold fidelity.
    * ``gamma``        -- coincidence factor of the spike trains (window ``delta``).
    * ``n_true``, ``n_pred``, ``count_err`` -- spike counts and absolute error.
    * ``latency_err_ms`` -- |first-spike latency error| (NaN if either is silent).
    """
    b, n = v_true.shape
    duration = n * dt
    out: dict[str, list[float]] = {k: [] for k in (
        "rmse_mv", "sub_rmse_mv", "gamma", "n_true", "n_pred", "count_err", "latency_err_ms")}
    for k in range(b):
        st, sp = spike_times(v_true[k], dt), spike_times(v_pred[k], dt)
        err = v_pred[k] - v_true[k]
        sub = ~(spike_mask(n, dt, st) | spike_mask(n, dt, sp))
        out["rmse_mv"].append(float(np.sqrt(np.mean(err**2))))
        out["sub_rmse_mv"].append(float(np.sqrt(np.mean(err[sub] ** 2))) if sub.any() else np.nan)
        out["gamma"].append(coincidence_factor(st, sp, delta, duration))
        out["n_true"].append(float(len(st)))
        out["n_pred"].append(float(len(sp)))
        out["count_err"].append(float(abs(len(st) - len(sp))))
        out["latency_err_ms"].append(
            float(abs(st[0] - sp[0])) if len(st) and len(sp) else np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def aggregate(m: dict[str, FloatArray]) -> dict[str, float]:
    """Dataset-level summary of :func:`trace_metrics`."""
    spiking = m["n_true"] > 0
    rate_true, rate_pred = m["n_true"], m["n_pred"]
    ss_res = float(np.sum((rate_true - rate_pred) ** 2))
    ss_tot = float(np.sum((rate_true - rate_true.mean()) ** 2))
    return {
        "rmse_mv_mean": float(np.mean(m["rmse_mv"])),
        "rmse_mv_median": float(np.median(m["rmse_mv"])),
        "sub_rmse_mv_median": float(np.nanmedian(m["sub_rmse_mv"])),
        "gamma_mean": float(np.nanmean(m["gamma"])),
        "gamma_median": float(np.nanmedian(m["gamma"])),
        "gamma_mean_spiking": float(np.nanmean(m["gamma"][spiking])) if spiking.any() else np.nan,
        "spike_count_exact": float(np.mean(m["count_err"] == 0)),
        "spike_count_mae": float(np.mean(m["count_err"])),
        "spike_count_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "latency_err_ms_median": float(np.nanmedian(m["latency_err_ms"]))
        if np.isfinite(m["latency_err_ms"]).any() else np.nan,
        "frac_spiking": float(np.mean(spiking)),
        "n_traces": float(len(spiking)),
    }


def summary_errors(v_true: FloatArray, v_pred: FloatArray, dt: float) -> dict[str, float]:
    """Errors of electrophysiological summary features computed from the traces."""
    st, sp = summary_table(v_true, dt), summary_table(v_pred, dt)
    out: dict[str, float] = {}
    for key in ("rate_hz", "first_latency_ms", "v_mean_sub_mv", "v_max_mv"):
        ok = np.isfinite(st[key]) & np.isfinite(sp[key])
        mae = float(np.mean(np.abs(st[key][ok] - sp[key][ok]))) if ok.any() else np.nan
        out[f"{key}_mae"] = mae
    return out


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def auroc(neg: FloatArray, pos: FloatArray) -> float:
    """Area under the ROC curve for separating ``pos`` (higher) from ``neg``."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    # Mann-Whitney U statistic with ties counted as 1/2.
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def spearman(x: FloatArray, y: FloatArray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def coverage(v_true: FloatArray, mean: FloatArray, std: FloatArray, k: float = 2.0) -> float:
    """Fraction of samples with ``|truth - mean| <= k * std``."""
    return float(np.mean(np.abs(v_true - mean) <= k * std))


def extrapolation_distance(theta: FloatArray, norm: Normalizer) -> FloatArray:
    """How far outside the training box a parameter vector is (box half-widths)."""
    th = norm.theta(theta).numpy().astype(np.float64)
    return np.asarray(np.max(np.clip(np.abs(th) - 1.0, 0.0, None), axis=1), dtype=np.float64)


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


def time_call(fn: Callable[[], object], repeats: int = 3) -> float:
    """Best-of-``repeats`` wall-clock time of ``fn()`` after one warm-up call."""
    fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def tile_inputs(ds: Dataset, n: int) -> tuple[FloatArray, FloatArray]:
    """The first ``n`` (theta, current) pairs of ``ds``, repeated as needed."""
    reps = int(np.ceil(n / len(ds)))
    return np.tile(ds.theta, (reps, 1))[:n], np.tile(ds.current, (reps, 1))[:n]


def speed_benchmark(
    cfg: Config,
    ds: Dataset,
    models: dict[str, Surrogate],
    ensemble: Ensemble,
    norm: Normalizer,
    log: Callable[[str], None] = lambda _: None,
) -> dict[str, dict[str, float]]:
    """Seconds per simulated trace for the reference solvers and the surrogates.

    The scipy reference integrates traces one after another on one core (the
    usual workflow). The numpy RK4 integrator and the surrogates process a
    batch at once; the surrogates are timed on the CPU with 1 thread and with
    the default number, and, when a CUDA device is available, also on the GPU
    (keys ending in ``_cuda``; GPU timings include the host-device copies).
    """
    base = HHParams(temperature=cfg.sim.temperature)
    dt = cfg.sim.dt_ms
    res: dict[str, dict[str, float]] = {}

    n_ref = min(cfg.eval.speed_n_scipy, len(ds))
    t0 = time.perf_counter()
    for k in range(n_ref):
        simulate_scipy(ds.theta[k], ds.current[k], dt, base, cfg.sim.rtol, cfg.sim.atol)
    res["scipy_lsoda"] = {"sec_per_trace": (time.perf_counter() - t0) / n_ref, "batch": 1.0,
                          "threads": 1.0}
    log(f"  scipy LSODA: {res['scipy_lsoda']['sec_per_trace'] * 1e3:.1f} ms/trace")

    big = max(cfg.eval.speed_batch_sizes)
    th_big, cur_big = tile_inputs(ds, big)
    t = time_call(lambda: simulate_batch(th_big, cur_big, dt, cfg.sim.rk4_substeps, base), 1)
    res["numpy_rk4_batch"] = {"sec_per_trace": t / big, "batch": float(big), "threads": 1.0}
    log(f"  numpy RK4 (batch {big}): {t / big * 1e3:.2f} ms/trace")

    cpu_models = {name: on_device(m, "cpu") for name, m in models.items()}
    cpu_ens = Ensemble([on_device(m, "cpu") for m in ensemble.members], norm)
    default_threads = torch.get_num_threads()
    for threads in sorted({1, default_threads}):
        torch.set_num_threads(threads)
        for name, model in cpu_models.items():
            for bs in cfg.eval.speed_batch_sizes:
                th, cur = tile_inputs(ds, bs)
                t = time_call(partial(predict_mv, model, norm, th, cur, bs))
                key = f"{name}_b{bs}_t{threads}"
                res[key] = {"sec_per_trace": t / bs, "batch": float(bs), "threads": float(threads)}
                log(f"  {key}: {t / bs * 1e3:.3f} ms/trace")
        t = time_call(lambda: cpu_ens.predict(th_big, cur_big, big))
        key = f"ensemble{len(ensemble.members)}_b{big}_t{threads}"
        res[key] = {"sec_per_trace": t / big, "batch": float(big), "threads": float(threads)}
        log(f"  {key}: {t / big * 1e3:.3f} ms/trace")
    torch.set_num_threads(default_threads)

    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        gpu_models = {name: on_device(m, gpu) for name, m in models.items()}
        gpu_ens = Ensemble([on_device(m, gpu) for m in ensemble.members], norm)
        for bs in cfg.eval.gpu_batch_sizes:
            th, cur = tile_inputs(ds, bs)
            for name, model in gpu_models.items():
                t = time_call(partial(predict_mv, model, norm, th, cur, bs))
                key = f"{name}_b{bs}_cuda"
                res[key] = {"sec_per_trace": t / bs, "batch": float(bs), "threads": 0.0}
                log(f"  {key}: {t / bs * 1e3:.4f} ms/trace")
            t = time_call(partial(gpu_ens.predict, th, cur, bs))
            key = f"ensemble{len(ensemble.members)}_b{bs}_cuda"
            res[key] = {"sec_per_trace": t / bs, "batch": float(bs), "threads": 0.0}
            log(f"  {key}: {t / bs * 1e3:.4f} ms/trace")

    ref = res["scipy_lsoda"]["sec_per_trace"]
    rk4 = res["numpy_rk4_batch"]["sec_per_trace"]
    for v in res.values():
        v["speedup_vs_scipy"] = ref / v["sec_per_trace"]
        v["speedup_vs_numpy_rk4"] = rk4 / v["sec_per_trace"]
    return res


def selection_speedups(
    cfg: Config, ds: Dataset, models: dict[str, Surrogate], norm: Normalizer, n_ref: int = 10
) -> dict[str, float]:
    """Quick batched-inference speed-up of each candidate vs scipy LSODA.

    Used for architecture selection; the full benchmark comes later.
    """
    base = HHParams(temperature=cfg.sim.temperature)
    n_ref = min(n_ref, len(ds))
    t0 = time.perf_counter()
    for k in range(n_ref):
        simulate_scipy(ds.theta[k], ds.current[k], cfg.sim.dt_ms, base, cfg.sim.rtol, cfg.sim.atol)
    ref = (time.perf_counter() - t0) / n_ref
    big = max(cfg.eval.speed_batch_sizes)
    th, cur = tile_inputs(ds, big)
    out = {}
    for name, model in models.items():
        t = time_call(partial(predict_mv, model, norm, th, cur, big), repeats=1)
        out[name] = ref / (t / big)
    return out


def reference_agreement(cfg: Config, ds: Dataset) -> dict[str, float]:
    """Discrepancy between the dataset integrator (RK4) and scipy LSODA."""
    n = min(cfg.eval.n_scipy_check, len(ds))
    base = HHParams(temperature=cfg.sim.temperature)
    ref = np.stack([
        simulate_scipy(ds.theta[k], ds.current[k], cfg.sim.dt_ms, base, 1e-9, 1e-11)[:, 0]
        for k in range(n)
    ])
    rk4 = ds.voltage[:n]
    gam = [
        coincidence_factor(spike_times(ref[k], ds.dt), spike_times(rk4[k], ds.dt),
                           cfg.eval.coincidence_window_ms, ref.shape[1] * ds.dt)
        for k in range(n)
    ]
    return {
        "n_traces": float(n),
        "max_abs_err_mv": float(np.max(np.abs(ref - rk4))),
        "rmse_mv": float(np.sqrt(np.mean((ref - rk4) ** 2))),
        "gamma_mean": float(np.mean(gam)),
    }


# ---------------------------------------------------------------------------
# f-I curves
# ---------------------------------------------------------------------------


def fi_curves(
    cfg: Config,
    thetas: FloatArray,
    ensemble: Ensemble,
    onset_ms: float = 10.0,
) -> dict[str, FloatArray]:
    """Firing rate during a long current step, truth vs twin, for each ``theta``."""
    n = cfg.sim.n_steps
    dt = cfg.sim.dt_ms
    amps = np.asarray(cfg.eval.fi_currents, dtype=np.float64)
    t = dt * np.arange(n)
    th = np.repeat(thetas, len(amps), axis=0)
    cur = np.repeat((t >= onset_ms)[None, :].astype(np.float64), len(th), axis=0)
    cur *= np.tile(amps, len(thetas))[:, None]
    true_v = simulate_batch(th, cur, dt, cfg.sim.rk4_substeps,
                            HHParams(temperature=cfg.sim.temperature))[0][:, :, 0]
    _, _, members = ensemble.predict(th, cur)
    window_s = (n * dt - onset_ms) / 1000.0

    def rate(v: FloatArray) -> FloatArray:
        return np.array([len(spike_times(r, dt)) / window_s for r in v])

    shape = (len(thetas), len(amps))
    member_rates = np.stack([rate(m).reshape(shape) for m in members])
    return {
        "amps": amps,
        "true": rate(true_v).reshape(shape),
        # Rates are computed per member and then averaged: averaging voltage
        # traces first would blur spikes on which members disagree.
        "pred": member_rates.mean(axis=0),
        "pred_member_std": member_rates.std(axis=0),
    }
