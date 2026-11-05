"""Figures for docs/RESULTS.md (matplotlib, headless)."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter
from numpy.typing import NDArray

from .hh import PARAM_NAMES

FloatArray = NDArray[np.float64]

# Okabe-Ito colour-blind-safe palette.
TRUE_C = "#000000"
TWIN_C = "#0072B2"
NODE_C = "#E69F00"
OOD_P_C = "#D55E00"
OOD_S_C = "#CC79A7"
ID_C = "#0072B2"
GREY = "#999999"

SPLIT_LABEL = {
    "test_id": "held-out params (in range)",
    "test_ood_param": "extrapolated params",
    "test_ood_stim": "novel stimuli",
    "val": "validation",
}
SPLIT_COLOR = {"test_id": ID_C, "test_ood_param": OOD_P_C, "test_ood_stim": OOD_S_C, "val": GREY}
PARAM_LABEL = {
    "g_na": r"$\bar g_{Na}$ (mS/cm$^2$)",
    "g_k": r"$\bar g_{K}$ (mS/cm$^2$)",
    "g_l": r"$g_{L}$ (mS/cm$^2$)",
    "c_m": r"$C_m$ ($\mu$F/cm$^2$)",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
    })


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _trace_panel(
    fig: Figure, spec: Any, t: FloatArray, current: FloatArray, v_true: FloatArray,
    mean: FloatArray, std: FloatArray, title: str, show_legend: bool,
) -> None:
    sub = spec.subgridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    ax_i: Axes = fig.add_subplot(sub[0])
    ax_v: Axes = fig.add_subplot(sub[1], sharex=ax_i)
    ax_i.plot(t, current, color=GREY, lw=0.8)
    ax_i.set_ylabel(r"I ($\mu$A/cm$^2$)", fontsize=7)
    ax_i.tick_params(labelbottom=False, labelsize=7)
    ax_i.set_title(title)
    ax_v.fill_between(t, mean - 2 * std, mean + 2 * std, color=TWIN_C, alpha=0.25, lw=0,
                      label=r"twin $\pm 2\sigma$")
    ax_v.plot(t, v_true, color=TRUE_C, lw=1.0, label="HH simulator")
    ax_v.plot(t, mean, color=TWIN_C, lw=0.9, ls="--", label="twin (ensemble mean)")
    ax_v.set_ylabel("V (mV)")
    ax_v.set_xlabel("time (ms)")
    ax_v.set_ylim(-95, 60)
    if show_legend:
        ax_v.legend(loc="upper right", ncol=1)


def plot_traces(path: Path, examples: list[dict[str, Any]], dt: float) -> None:
    """Grid of true-vs-surrogate traces; each example dict holds arrays + title."""
    _style()
    n = len(examples)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(11, 3.1 * rows))
    gs = fig.add_gridspec(rows, cols, hspace=0.45, wspace=0.22)
    for k, ex in enumerate(examples):
        t = dt * np.arange(1, len(ex["v_true"]) + 1, dtype=np.float64)
        _trace_panel(fig, gs[k // cols, k % cols], t, ex["current"], ex["v_true"], ex["mean"],
                     ex["std"], ex["title"], show_legend=(k == 0))
    _save(fig, path)


def _binned_median(
    x: FloatArray, y: FloatArray, edges: FloatArray
) -> tuple[FloatArray, FloatArray]:
    centers, meds = [], []
    for a, b in itertools.pairwise(edges):
        sel = (x >= a) & (x < b) & np.isfinite(y)
        if sel.sum() >= 5:
            centers.append(0.5 * (a + b))
            meds.append(float(np.median(y[sel])))
    return np.asarray(centers, dtype=np.float64), np.asarray(meds, dtype=np.float64)


def plot_error_vs_parameter(
    path: Path,
    per_split: dict[str, dict[str, FloatArray]],
    train_box: dict[str, tuple[float, float]],
) -> None:
    """Trace error and predictive uncertainty vs each parameter and vs the
    distance outside the training box (lines = binned medians)."""
    _style()
    n = len(PARAM_NAMES) + 1
    fig, axes = plt.subplots(2, n, figsize=(15, 5.6), sharey="row")
    splits = ("test_id", "test_ood_param")
    for j in range(n):
        for row, key, ylabel in ((0, "rmse_mv", "trace RMSE (mV)"),
                                 (1, "u_mv", r"predictive $\bar\sigma$ (mV)")):
            ax = axes[row, j]
            if j < len(PARAM_NAMES):
                name = PARAM_NAMES[j]
                lo, hi = train_box[name]
                ax.axvspan(lo, hi, color=GREY, alpha=0.18, lw=0,
                           label="training range" if j == 0 and row == 0 else None)
                xs = {s: per_split[s]["theta"][:, j] for s in splits}
                xlabel = PARAM_LABEL[name]
            else:
                xs = {s: per_split[s]["extrap"] for s in splits}
                xlabel = "distance outside training box\n(box half-widths, max over params)"
            allx = np.concatenate(list(xs.values()))
            edges = np.linspace(allx.min(), allx.max() + 1e-9, 9)
            for split in splits:
                ax.scatter(xs[split], per_split[split][key], s=6, alpha=0.45, lw=0,
                           color=SPLIT_COLOR[split],
                           label=SPLIT_LABEL[split] if j == 0 and row == 0 else None)
            pooled_y = np.concatenate([per_split[s][key] for s in splits])
            cx, my = _binned_median(allx, pooled_y, edges)
            ax.plot(cx, my, color=TRUE_C, lw=1.5, marker="o", ms=3,
                    label="binned median" if j == 0 and row == 0 else None)
            ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel(xlabel)
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Accuracy and uncertainty across parameter space", y=1.07)
    fig.tight_layout()
    _save(fig, path)


def plot_speedup(path: Path, rows: list[tuple[str, float, float]]) -> None:
    """Horizontal bars of speed-up vs scipy LSODA; rows = (label, speedup, ms/trace)."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.2))
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    colors = [GREY if ("scipy" in lab or "numpy" in lab) else
              (NODE_C if "neural ODE" in lab else TWIN_C) for lab in labels]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.axvline(1.0, color=TRUE_C, lw=0.8)
    for k, (_, sp, ms) in enumerate(rows):
        ax.text(sp * 1.08, k, f"{sp:,.0f}x  ({ms:.3g} ms/trace)" if sp >= 10
                else f"{sp:.2g}x  ({ms:.3g} ms/trace)", va="center", fontsize=8)
    ax.set_xlim(right=vals.max() * 12)
    ax.set_xlabel("speed-up vs scipy LSODA (per trace, log scale)")
    ax.set_title("Wall-clock cost of simulating one 100 ms trace (CPU)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, path)


def plot_uncertainty(
    path: Path, per_split: dict[str, dict[str, FloatArray]], threshold: float
) -> None:
    """Uncertainty vs error, and uncertainty distributions per split."""
    _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("test_id", "test_ood_stim", "test_ood_param"):
        d = per_split[split]
        a.scatter(d["u_mv"], d["rmse_mv"], s=8, alpha=0.6, lw=0, color=SPLIT_COLOR[split],
                  label=SPLIT_LABEL[split])
    a.axvline(threshold, color=TRUE_C, ls=":", lw=1, label="extrapolation flag threshold")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel(r"predictive uncertainty $\bar\sigma$ (mV, ensemble spread)")
    a.set_ylabel("actual trace RMSE (mV)")
    a.set_title("Uncertainty tracks error")
    a.legend(loc="upper left")
    allu = np.concatenate([per_split[s]["u_mv"] for s in per_split])
    bins = np.logspace(np.log10(max(allu.min(), 1e-3)), np.log10(allu.max()), 30)
    for split in ("test_id", "test_ood_stim", "test_ood_param"):
        b.hist(per_split[split]["u_mv"], bins=bins, histtype="step", lw=1.5,
               color=SPLIT_COLOR[split], label=SPLIT_LABEL[split], density=True)
    b.axvline(threshold, color=TRUE_C, ls=":", lw=1)
    b.set_xscale("log")
    b.set_xlabel(r"predictive uncertainty $\bar\sigma$ (mV)")
    b.set_ylabel("density")
    b.set_title("The twin is less certain when extrapolating")
    b.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, path)


def plot_model_comparison(
    path: Path,
    curves: dict[str, tuple[list[float], list[float]]],
    scores: dict[str, dict[str, float]],
    example: dict[str, FloatArray],
    dt: float,
) -> None:
    """Learning curves, validation metrics and an example: GRU vs neural ODE."""
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6),
                             gridspec_kw={"width_ratios": [1.1, 0.9, 0.55, 1.6]})
    colors = {"gru": TWIN_C, "node": NODE_C}
    names = {"gru": "GRU", "node": "neural ODE"}
    for kind, (tr, va) in curves.items():
        axes[0].plot(np.arange(1, len(va) + 1), va, color=colors[kind], label=f"{names[kind]} val")
        axes[0].plot(np.arange(1, len(tr) + 1), tr, color=colors[kind], ls=":", alpha=0.7,
                     label=f"{names[kind]} train")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE (normalised)")
    axes[0].set_title("Learning curves\n(NODE train loss = short windows)")
    axes[0].legend()

    w = 0.38
    for ax, metrics, title in (
        (axes[1], [("gamma_mean", r"mean $\Gamma$"), ("spike_count_exact", "exact count")],
         "Validation spike metrics\n(higher is better)"),
        (axes[2], [("rmse_mv_median", "median RMSE")], "Voltage error (mV)\n(lower is better)"),
    ):
        x = np.arange(len(metrics))
        for k, kind in enumerate(scores):
            vals = [scores[kind][m] for m, _ in metrics]
            bars = ax.bar(x + (k - 0.5) * w, vals, w, color=colors[kind], label=names[kind])
            ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.set_xticks(x, [lab for _, lab in metrics], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[1].set_ylim(0, 1.3)
    axes[1].legend(loc="upper center", ncol=2)

    t = dt * np.arange(1, len(example["v_true"]) + 1)
    axes[3].plot(t, example["v_true"], color=TRUE_C, lw=1.0, label="HH simulator")
    axes[3].plot(t, example["gru"], color=TWIN_C, lw=0.9, ls="--", label="GRU")
    axes[3].plot(t, example["node"], color=NODE_C, lw=0.9, ls="-.", label="neural ODE")
    axes[3].set_xlabel("time (ms)")
    axes[3].set_ylabel("V (mV)")
    axes[3].set_title("Typical validation trace (median GRU error)")
    axes[3].legend(loc="upper right")
    fig.tight_layout()
    _save(fig, path)


def plot_fi(path: Path, fi: dict[str, FloatArray], labels: list[str]) -> None:
    """f-I curves: truth vs twin (mean +/- member std)."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = [TWIN_C, OOD_P_C, "#009E73", OOD_S_C]
    for k, lab in enumerate(labels):
        c = palette[k % len(palette)]
        ax.plot(fi["amps"], fi["true"][k], color=c, lw=1.6, marker="o", ms=3,
                label=f"{lab}: simulator")
        ax.errorbar(fi["amps"], fi["pred"][k], yerr=fi["pred_member_std"][k], color=c, lw=1.0,
                    ls="--", marker="s", ms=3, mfc="white", capsize=2, label=f"{lab}: twin")
    ax.set_xlabel(r"step current ($\mu$A/cm$^2$)")
    ax.set_ylabel("firing rate during step (Hz)")
    ax.set_title("f-I curves (90 ms steps)")
    ax.legend(fontsize=7, ncol=1)
    fig.tight_layout()
    _save(fig, path)


def plot_failure_modes(
    path: Path,
    by_family: dict[str, FloatArray],
    err_vs_time: dict[str, FloatArray],
    dt: float,
    worst: dict[str, Any],
) -> None:
    """Gamma by stimulus family, error growth over time, and the worst ID trace."""
    _style()
    fig = plt.figure(figsize=(13, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.4], wspace=0.3)
    a = fig.add_subplot(gs[0])
    fams = list(by_family)
    a.boxplot([by_family[f] for f in fams], showfliers=True, flierprops={"ms": 2})
    a.set_xticks(np.arange(1, len(fams) + 1), fams, rotation=30, fontsize=8)
    a.set_ylabel(r"coincidence factor $\Gamma$")
    a.set_title("Spike-timing accuracy by stimulus\n(* = never seen in training)")
    a.set_ylim(-0.3, 1.05)

    b = fig.add_subplot(gs[1])
    for split, curve in err_vs_time.items():
        t = dt * np.arange(1, len(curve) + 1)
        b.plot(t, curve, color=SPLIT_COLOR[split], lw=1, label=SPLIT_LABEL[split])
    b.set_xlabel("time (ms)")
    b.set_ylabel(r"mean $|V_{twin} - V_{true}|$ (mV)")
    b.set_title("Error vs time since stimulus start")
    b.legend(loc="upper left")

    sub = gs[2].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    t = dt * np.arange(1, len(worst["v_true"]) + 1)
    ai = fig.add_subplot(sub[0])
    ai.plot(t, worst["current"], color=GREY, lw=0.8)
    ai.tick_params(labelbottom=False, labelsize=7)
    ai.set_title(worst["title"])
    av = fig.add_subplot(sub[1], sharex=ai)
    av.fill_between(t, worst["mean"] - 2 * worst["std"], worst["mean"] + 2 * worst["std"],
                    color=TWIN_C, alpha=0.25, lw=0)
    av.plot(t, worst["v_true"], color=TRUE_C, lw=1.0, label="HH simulator")
    av.plot(t, worst["mean"], color=TWIN_C, lw=0.9, ls="--", label="twin")
    av.set_xlabel("time (ms)")
    av.set_ylabel("V (mV)")
    av.legend(loc="upper right")
    _save(fig, path)


def plot_patch_sweep(path: Path, result: dict[str, Any]) -> None:
    """Accuracy and speed-up of the GRU against its time step (patch length)."""
    _style()
    rows = result["rows"]
    x = np.array([r["step_ms"] for r in rows], dtype=np.float64)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for split, label, color in (("val", "validation", GREY),
                                ("test_id", SPLIT_LABEL["test_id"], ID_C),
                                ("test_ood_param", SPLIT_LABEL["test_ood_param"], OOD_P_C),
                                ("test_ood_stim", SPLIT_LABEL["test_ood_stim"], OOD_S_C)):
        a.plot(x, [r[split]["gamma_mean"] for r in rows], marker="o", color=color, label=label)
    a.set_xscale("log")
    a.set_xticks(x, [f"{v:g}" for v in x])
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel("GRU time step = patch length (ms)")
    a.set_ylabel(r"mean coincidence factor $\Gamma$")
    a.set_title("Accuracy vs time step")
    a.legend(loc="center right")
    keys = list(rows[0]["speedup_vs_scipy"])
    palette = {"cpu": TWIN_C, "cuda": NODE_C}
    styles = ["-", "--", ":", "-."]
    for k in keys:
        dev, bs = k.split("_b")
        same_dev = [kk for kk in keys if kk.startswith(dev)]
        b.plot(x, [r["speedup_vs_scipy"][k] for r in rows], marker="o", color=palette[dev],
               ls=styles[same_dev.index(k) % len(styles)],
               label=f"{'GPU' if dev == 'cuda' else 'CPU'}, batch {int(bs):,}")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xticks(x, [f"{v:g}" for v in x])
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xlabel("GRU time step = patch length (ms)")
    b.set_ylabel("speed-up vs scipy LSODA (single model)")
    gpu = result.get("gpu")
    b.set_title("Inference speed vs time step" + (f"\n(GPU: {gpu})" if gpu else " (CPU only)"))
    b.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)
