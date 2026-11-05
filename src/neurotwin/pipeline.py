"""End-to-end experiment: data -> model selection -> ensemble -> evaluation -> report."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import platform
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import plots
from .config import Config
from .data import Dataset, build_datasets
from .evaluate import (
    aggregate,
    auroc,
    coverage,
    extrapolation_distance,
    fi_curves,
    reference_agreement,
    selection_speedups,
    spearman,
    speed_benchmark,
    summary_errors,
    trace_metrics,
)
from .hh import PARAM_NAMES
from .models import (
    Ensemble,
    Normalizer,
    Surrogate,
    build_model,
    count_parameters,
    cpu_state_dict,
    device_label,
    predict_mv,
    resolve_device,
)
from .train import TrainResult, train_gru, train_node

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
TEST_SPLITS = ("test_id", "test_ood_param", "test_ood_stim")
KIND_NAME = {"gru": "GRU", "node": "neural ODE"}


def _train(kind: str, data: dict[str, Dataset], norm: Normalizer, cfg: Config, seed: int,
           log: Callable[[str], None]) -> TrainResult:
    device = resolve_device(cfg.device)
    if kind == "gru":
        return train_gru(data["train"], data["val"], norm, cfg.gru, seed, log, device=device)
    return train_node(data["train"], data["val"], norm, cfg.node, seed, log, device=device)


def _print(msg: str) -> None:
    print(msg, flush=True)


TrainJob = tuple[str, dict[str, Dataset], Normalizer, Config, int, int]


def _train_job(job: TrainJob) -> tuple[dict[str, object], dict[str, Any], TrainResult]:
    """Worker-process entry point: train one model, return it as plain data."""
    kind, data, norm, cfg, seed, threads = job
    torch.set_num_threads(threads)
    res = _train(kind, data, norm, cfg, seed, _print)
    state = {k: v.numpy() for k, v in cpu_state_dict(res.model).items()}
    return res.model.config_dict(), state, TrainResult(
        model=build_model(kind, res.model.config_dict()), train_loss=res.train_loss,
        val_loss=res.val_loss, best_epoch=res.best_epoch, seconds=res.seconds)


def _model_cache_path(cache: Path, kind: str, seed: int, cfg: Config) -> Path:
    """Checkpoint path keyed by everything that determines the trained model."""
    arch = cfg.gru if kind == "gru" else cfg.node
    parts: tuple[object, ...] = (kind, seed, arch, cfg.seed, cfg.sim, cfg.params, cfg.stimulus,
                                 cfg.data.n_train, cfg.data.n_val)
    if resolve_device(cfg.device).type == "cuda":
        parts += ("cuda",)  # GPU training is numerically different from CPU training
    key = repr(parts)
    return cache / f"{kind}-seed{seed}-{hashlib.sha1(key.encode()).hexdigest()[:10]}.pt"


def _save_result(path: Path, res: TrainResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": res.model.kind, "config": res.model.config_dict(),
                "state_dict": cpu_state_dict(res.model), "train_loss": res.train_loss,
                "val_loss": res.val_loss, "best_epoch": res.best_epoch,
                "seconds": res.seconds}, path)


def _load_result(path: Path) -> TrainResult:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["kind"], ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return TrainResult(model=model, train_loss=list(ckpt["train_loss"]),
                       val_loss=list(ckpt["val_loss"]), best_epoch=int(ckpt["best_epoch"]),
                       seconds=float(ckpt["seconds"]))


def train_many(
    jobs: list[tuple[str, int]],
    data: dict[str, Dataset],
    norm: Normalizer,
    cfg: Config,
    log: Callable[[str], None],
    cache: Path | None = None,
    job_cfgs: list[Config] | None = None,
) -> list[TrainResult]:
    """Train ``(kind, seed)`` jobs, in ``cfg.parallel_jobs`` processes if > 1.

    Small recurrent networks use several cores poorly, so independent models
    (candidates, ensemble members) train faster side by side with one or two
    threads each than one after another with all threads. Trained models are
    cached in ``cache`` (if given) and reused by later runs of the same config.
    ``job_cfgs`` optionally gives each job its own config (e.g. a patch sweep).
    The returned models live on ``cfg.device``.
    """
    cfgs = job_cfgs if job_cfgs is not None else [cfg] * len(jobs)
    if len(cfgs) != len(jobs):
        raise ValueError("job_cfgs must have one config per job")
    results: dict[int, TrainResult] = {}
    todo: list[int] = []
    for i, (kind, seed) in enumerate(jobs):
        path = _model_cache_path(cache, kind, seed, cfgs[i]) if cache else None
        if path is not None and path.exists():
            results[i] = _load_result(path)
            log(f"  loaded cached {KIND_NAME[kind]} (seed {seed}) from {path}")
        else:
            todo.append(i)
    n_proc = min(cfg.parallel_jobs, len(todo))
    if n_proc <= 1:
        for i in todo:
            kind, seed = jobs[i]
            results[i] = _train(kind, data, norm, cfgs[i], seed, log)
    else:
        total = cfg.torch_threads or torch.get_num_threads()
        threads = max(1, total // n_proc)
        sub = {"train": data["train"], "val": data["val"]}
        log(f"  training {len(todo)} models in {n_proc} processes x {threads} threads")
        ctx = multiprocessing.get_context("spawn")
        args = [(jobs[i][0], sub, norm, cfgs[i], jobs[i][1], threads) for i in todo]
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=ctx) as pool:
            outs = list(pool.map(_train_job, args))
        for i, (_, state, res) in zip(todo, outs, strict=True):
            res.model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
            res.model.eval()
            results[i] = res
    if cache is not None:
        for i in todo:
            _save_result(_model_cache_path(cache, *jobs[i], cfgs[i]), results[i])
    device = resolve_device(cfg.device)
    for res in results.values():
        res.model.to(device).eval()
    return [results[i] for i in range(len(jobs))]


GAMMA_TIE = 0.01
SELECTION_RULE = (
    "candidates meeting the speed-up requirement (all, if none does) whose validation "
    "mean Gamma is within {tie} of the best are equally accurate; among them the "
    "fastest is chosen"
)


def select_model(
    scores: dict[str, dict[str, float]],
    speedups: dict[str, float] | None = None,
    min_speedup: float = 0.0,
    tie: float = GAMMA_TIE,
) -> str:
    """Pick the architecture for the twin.

    1. Keep the candidates whose batched inference is at least ``min_speedup``
       times faster than the reference solver (all candidates if none is).
    2. Treat candidates whose validation mean Gamma is within ``tie`` of the
       best as equally accurate.
    3. Among those, take the fastest (the twin exists to be fast); without
       speed information, take the lowest median voltage RMSE.
    """
    eligible = [k for k in scores if speedups is None or speedups[k] >= min_speedup]
    eligible = eligible or list(scores)
    best_gamma = max(scores[k]["gamma_mean"] for k in eligible)
    tied = [k for k in eligible if best_gamma - scores[k]["gamma_mean"] < tie]
    if speedups is not None:
        return max(tied, key=lambda k: speedups[k])
    return min(tied, key=lambda k: scores[k]["rmse_mv_median"])


def save_twin(path: Path, members: list[Surrogate], norm: Normalizer) -> None:
    """Persist the ensemble (one file per member) and the normaliser."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "normalizer.json").write_text(json.dumps(norm.to_dict(), indent=2))
    for k, m in enumerate(members):
        torch.save({"kind": m.kind, "config": m.config_dict(), "state_dict": cpu_state_dict(m)},
                   path / f"member{k}.pt")


def load_twin(path: str | Path, device: str = "cpu") -> Ensemble:
    """Load an ensemble saved by :func:`save_twin` onto ``device`` ("auto" allowed)."""
    path = Path(path)
    norm = Normalizer(**json.loads((path / "normalizer.json").read_text()))
    members: list[Surrogate] = []
    for f in sorted(path.glob("member*.pt")):
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
        m = build_model(ckpt["kind"], ckpt["config"])
        m.load_state_dict(ckpt["state_dict"])
        members.append(m.to(resolve_device(device)).eval())
    return Ensemble(members, norm)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr[np.isfinite(arr)].mean()) if np.isfinite(arr).any() else float("nan")


def _r(x: float, nd: int = 3) -> float:
    return float(round(x, nd)) if np.isfinite(x) else float("nan")


def run(
    cfg: Config,
    out: Path,
    figdir: Path | None = None,
    log: Callable[[str], None] = print,
    make_figures: bool = True,
) -> dict[str, Any]:
    """Run the whole experiment and write metrics, report, checkpoints and figures."""
    t_start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    figdir = figdir or out / "figures"
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    dt = cfg.sim.dt_ms
    device = resolve_device(cfg.device)

    # 1. Data ---------------------------------------------------------------
    log(f"[1/6] simulating datasets ({cfg.name}; networks on {device_label(device)})")
    t0 = time.perf_counter()
    data = build_datasets(cfg, out / "data")
    t_data = time.perf_counter() - t0
    log(f"  {', '.join(f'{k}={len(v)}' for k, v in data.items())} traces "
        f"({t_data:.1f} s; stiff LSODA fallbacks: "
        f"{', '.join(f'{k}={v.n_fallback}' for k, v in data.items())})")
    lo, hi = cfg.params.bounds("train")
    norm = Normalizer.fit(lo, hi, data["train"].current, data["train"].states)
    val = data["val"]

    # 2. Candidate architectures, selection on validation -------------------
    log("[2/6] training candidate surrogates (GRU, neural ODE)")
    cand: dict[str, TrainResult] = {}
    val_scores: dict[str, dict[str, float]] = {}
    val_pred: dict[str, FloatArray] = {}
    kinds = ("gru", "node")
    trained = train_many([(k, cfg.seed * 1000 + 1) for k in kinds], data, norm, cfg, log,
                         cache=out / "models")
    for kind, res in zip(kinds, trained, strict=True):
        cand[kind] = res
        val_pred[kind] = predict_mv(cand[kind].model, norm, val.theta, val.current)
        val_scores[kind] = aggregate(trace_metrics(
            val.voltage, val_pred[kind], dt, cfg.eval.coincidence_window_ms))
        log(f"  {KIND_NAME[kind]}: val Gamma={val_scores[kind]['gamma_mean']:.3f} "
            f"RMSE={val_scores[kind]['rmse_mv_median']:.2f} mV "
            f"({cand[kind].seconds:.0f} s, {count_parameters(cand[kind].model)} params)")
    sel_speed = selection_speedups(cfg, val, {k: cand[k].model for k in kinds}, norm)
    for kind in kinds:
        log(f"  {KIND_NAME[kind]}: batched inference {sel_speed[kind]:.0f}x faster than scipy "
            f"(requirement: >= {cfg.eval.min_speedup:.0f}x)")
    winner = select_model(val_scores, sel_speed, cfg.eval.min_speedup)
    log(f"  selected: {KIND_NAME[winner]}")

    # 3. Deep ensemble of the winner ----------------------------------------
    log(f"[3/6] training ensemble of {cfg.ensemble.members} x {KIND_NAME[winner]}")
    extra = train_many([(winner, cfg.seed * 1000 + 1 + k) for k in range(1, cfg.ensemble.members)],
                       data, norm, cfg, log, cache=out / "models")
    members = [cand[winner].model] + [r.model for r in extra]
    ens_seconds = [cand[winner].seconds] + [r.seconds for r in extra]
    ens = Ensemble(members, norm)
    save_twin(out / "twin", members, norm)

    # 4. Accuracy + uncertainty on held-out splits --------------------------
    log("[4/6] evaluating on held-out splits")
    per_split: dict[str, dict[str, FloatArray]] = {}
    split_metrics: dict[str, Any] = {}
    preds: dict[str, tuple[FloatArray, FloatArray, FloatArray]] = {}
    for split in ("val", *TEST_SPLITS):
        ds = data[split]
        mean, std, mem = ens.predict(ds.theta, ds.current)
        preds[split] = (mean, std, mem)
        tm = trace_metrics(ds.voltage, mean, dt, cfg.eval.coincidence_window_ms)
        single = [aggregate(trace_metrics(ds.voltage, m, dt, cfg.eval.coincidence_window_ms))
                  for m in mem]
        u = std.mean(axis=1)
        per_split[split] = {**tm, "u_mv": u, "theta": ds.theta,
                            "extrap": extrapolation_distance(ds.theta, norm)}
        split_metrics[split] = {
            "ensemble": {k: _r(v) for k, v in aggregate(tm).items()},
            "single_member_mean": {k: _r(_nanmean([s[k] for s in single]))
                                   for k in single[0]},
            "summary_feature_errors": {k: _r(v) for k, v in
                                       summary_errors(ds.voltage, mean, dt).items()},
            "coverage_2sigma": _r(coverage(ds.voltage, mean, std)),
            "uncertainty_mv_median": _r(float(np.median(u))),
            "spearman_uncertainty_vs_rmse": _r(spearman(u, tm["rmse_mv"])),
        }
        e = split_metrics[split]["ensemble"]
        log(f"  {split}: Gamma={e['gamma_mean']:.3f} RMSE(median)={e['rmse_mv_median']:.2f} mV "
            f"count-exact={e['spike_count_exact']:.2f} u(median)={np.median(u):.2f} mV")

    # The rejected architecture on the in-distribution test set (one model each).
    arch_test: dict[str, dict[str, float]] = {}
    for kind in ("gru", "node"):
        p = predict_mv(cand[kind].model, norm, data["test_id"].theta, data["test_id"].current)
        arch_test[kind] = {k: _r(v) for k, v in aggregate(trace_metrics(
            data["test_id"].voltage, p, dt, cfg.eval.coincidence_window_ms)).items()}

    thr = float(np.quantile(per_split["val"]["u_mv"], cfg.eval.flag_quantile))
    u_id = per_split["test_id"]["u_mv"]
    pooled_u = np.concatenate([per_split[s]["u_mv"] for s in TEST_SPLITS])
    pooled_rmse = np.concatenate([per_split[s]["rmse_mv"] for s in TEST_SPLITS])
    pooled_gamma = np.concatenate([per_split[s]["gamma"] for s in TEST_SPLITS])
    flagged = pooled_u > thr

    def combined(s: str) -> float:
        """Flag = parameters outside the training box OR ensemble spread > threshold."""
        d = per_split[s]
        return float(np.mean((d["extrap"] > 0) | (d["u_mv"] > thr)))

    unc = {
        "flag_threshold_mv": _r(thr),
        "flag_rate": {s: _r(float(np.mean(per_split[s]["u_mv"] > thr))) for s in TEST_SPLITS},
        "flag_rate_with_range_check": {s: _r(combined(s)) for s in TEST_SPLITS},
        "auroc_id_vs_ood_param": _r(auroc(u_id, per_split["test_ood_param"]["u_mv"])),
        "auroc_id_vs_ood_stim": _r(auroc(u_id, per_split["test_ood_stim"]["u_mv"])),
        "spearman_u_vs_rmse_pooled": _r(spearman(pooled_u, pooled_rmse)),
        "spearman_u_vs_extrapolation_distance": _r(spearman(
            per_split["test_ood_param"]["u_mv"], per_split["test_ood_param"]["extrap"])),
        "gamma_mean_flagged": _r(float(np.nanmean(pooled_gamma[flagged])))
        if flagged.any() else float("nan"),
        "gamma_mean_unflagged": _r(float(np.nanmean(pooled_gamma[~flagged])))
        if (~flagged).any() else float("nan"),
        "rmse_median_flagged": _r(float(np.median(pooled_rmse[flagged])))
        if flagged.any() else float("nan"),
        "rmse_median_unflagged": _r(float(np.median(pooled_rmse[~flagged])))
        if (~flagged).any() else float("nan"),
    }
    log(f"  uncertainty: AUROC(ID vs OOD params)={unc['auroc_id_vs_ood_param']:.3f} "
        f"AUROC(ID vs OOD stim)={unc['auroc_id_vs_ood_stim']:.3f} "
        f"flag rates={unc['flag_rate']}")

    by_family: dict[str, FloatArray] = {}
    for split in ("test_id", "test_ood_stim"):
        fam = data[split].family
        for f in sorted(set(fam.tolist())):
            label = f if split == "test_id" else f"{f}*"
            by_family[label] = per_split[split]["gamma"][fam == f]
    family_gamma = {k: _r(float(np.nanmean(v))) for k, v in by_family.items()}

    # 5. Speed + reference agreement ----------------------------------------
    log("[5/6] benchmarking speed")
    models = {KIND_NAME[k]: cand[k].model for k in ("gru", "node")}
    speed = speed_benchmark(cfg, data["test_id"], models, ens, norm, log)
    agreement = reference_agreement(cfg, data["test_id"])
    log(f"  RK4 dataset integrator vs LSODA(rtol=1e-9): max |dV|="
        f"{agreement['max_abs_err_mv']:.2e} mV, Gamma={agreement['gamma_mean']:.3f}")

    # f-I curves for the default HH neuron and two corners of the box.
    fi_thetas = np.array([[120.0, 36.0, 0.3, 1.0], [lo[0], hi[1], hi[2], lo[3]],
                          [hi[0], lo[1], lo[2], hi[3]]])
    fi_labels = ["classic HH", "low gNa / high gK", "high gNa / low gK"]
    fi = fi_curves(cfg, fi_thetas, ens)
    fi_mae = float(np.mean(np.abs(fi["true"] - fi["pred"])))

    # 6. Report + figures ----------------------------------------------------
    log("[6/6] writing metrics, report and figures")
    metrics: dict[str, Any] = {
        "config": asdict(cfg),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "cpu_threads": torch.get_num_threads(),
                        "machine": platform.machine(), "device": device_label(device)},
        "data_seconds": _r(t_data, 1),
        "lsoda_fallbacks": {k: v.n_fallback for k, v in data.items()},
        "selection": {
            "winner": winner,
            "rule": SELECTION_RULE.format(tie=GAMMA_TIE)
            + f" (requirement: batched speed-up >= {cfg.eval.min_speedup:g}x vs scipy LSODA)",
            "speedup_vs_scipy": {k: _r(v, 1) for k, v in sel_speed.items()},
            "validation": {k: {m: _r(v) for m, v in s.items()} for k, s in val_scores.items()},
            "test_id_single_model": arch_test,
            "train_seconds": {k: _r(r.seconds, 1) for k, r in cand.items()},
            "n_parameters": {k: count_parameters(r.model) for k, r in cand.items()},
            "best_epoch": {k: r.best_epoch for k, r in cand.items()},
        },
        "ensemble": {"members": len(members), "train_seconds": [_r(s, 1) for s in ens_seconds]},
        "splits": split_metrics,
        "family_gamma": family_gamma,
        "uncertainty": unc,
        "speed": {k: {m: _r(v, 6) for m, v in d.items()} for k, d in speed.items()},
        "reference_agreement": {k: _r(v, 6) for k, v in agreement.items()},
        "fi": {"labels": fi_labels, "thetas": fi_thetas.tolist(),
               "amps": fi["amps"].tolist(), "true_hz": fi["true"].tolist(),
               "twin_hz": fi["pred"].round(2).tolist(), "mae_hz": _r(fi_mae, 2)},
        "total_seconds": 0.0,
    }

    if make_figures:
        _figures(figdir, cfg, data, preds, per_split, cand, val_scores, val_pred, speed, thr,
                 fi, fi_labels, by_family, winner)
    metrics["total_seconds"] = _r(time.perf_counter() - t_start, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True))
    (out / "report.md").write_text(render_report(metrics))
    log(f"done in {metrics['total_seconds']:.0f} s -> {out}")
    return metrics


def _pick(scores: FloatArray, q: float) -> int:
    """Index of the trace whose score is closest to the ``q`` quantile."""
    finite = np.where(np.isfinite(scores), scores, np.nanmedian(scores))
    return int(np.argmin(np.abs(finite - np.quantile(finite, q))))


def _figures(
    figdir: Path, cfg: Config, data: dict[str, Dataset],
    preds: dict[str, tuple[FloatArray, FloatArray, FloatArray]],
    per_split: dict[str, dict[str, FloatArray]], cand: dict[str, TrainResult],
    val_scores: dict[str, dict[str, float]], val_pred: dict[str, FloatArray],
    speed: dict[str, dict[str, float]], thr: float, fi: dict[str, FloatArray],
    fi_labels: list[str], by_family: dict[str, FloatArray], winner: str,
) -> None:
    dt = cfg.sim.dt_ms
    examples: list[dict[str, Any]] = []
    for split in ("test_id", "test_ood_stim", "test_ood_param"):
        ds = data[split]
        mean, std, _ = preds[split]
        rmse = per_split[split]["rmse_mv"]
        # Quantile of the trace error: 0.5 = a typical case, 0.9 = a bad case.
        for q in (0.5, 0.9):
            k = _pick(rmse, q)
            g = per_split[split]["gamma"][k]
            tag = "typical" if q == 0.5 else "90th-pct error"
            examples.append({
                "current": ds.current[k], "v_true": ds.voltage[k], "mean": mean[k], "std": std[k],
                "title": f"{plots.SPLIT_LABEL[split]} ({tag}): {ds.family[k]}, "
                         f"RMSE {rmse[k]:.1f} mV, $\\Gamma$={g:.2f}",
            })
    plots.plot_traces(figdir / "traces_true_vs_surrogate.png", examples, dt)

    plots.plot_error_vs_parameter(figdir / "error_vs_parameter.png", per_split,
                                  cfg.params.train_box)

    big = max(cfg.eval.speed_batch_sizes)
    rows: list[tuple[str, float, float]] = []

    def add(label: str, key: str) -> None:
        if key in speed:
            rows.append((label, speed[key]["speedup_vs_scipy"], 1e3 * speed[key]["sec_per_trace"]))

    threads = int(max(v["threads"] for v in speed.values()))
    add("scipy LSODA (1 trace at a time)", "scipy_lsoda")
    add(f"numpy RK4, batch {big}", "numpy_rk4_batch")
    add(f"neural ODE, batch {big}, {threads} threads", f"neural ODE_b{big}_t{threads}")
    add("GRU, batch 1, 1 thread", "GRU_b1_t1")
    add(f"GRU, batch {big}, 1 thread", f"GRU_b{big}_t1")
    add(f"GRU, batch {big}, {threads} threads", f"GRU_b{big}_t{threads}")
    ens_key = next((k for k in speed if k.startswith("ensemble") and k.endswith(f"t{threads}")),
                   None)
    if ens_key:
        add(f"{KIND_NAME[winner]} ensemble x{cfg.ensemble.members}, batch {big}, {threads} threads",
            ens_key)
    plots.plot_speedup(figdir / "speedup.png", rows)

    plots.plot_uncertainty(figdir / "uncertainty.png", per_split, thr)

    vk = _pick(per_split["val"]["rmse_mv"], 0.5)
    vds = data["val"]
    plots.plot_model_comparison(
        figdir / "model_comparison.png",
        {k: (r.train_loss, r.val_loss) for k, r in cand.items()},
        val_scores,
        {"v_true": vds.voltage[vk], "gru": val_pred["gru"][vk], "node": val_pred["node"][vk]},
        dt,
    )
    plots.plot_fi(figdir / "fi_curves.png", fi, fi_labels)

    err_t = {s: np.abs(preds[s][0] - data[s].voltage).mean(axis=0) for s in TEST_SPLITS}
    wk = int(np.nanargmax(per_split["test_id"]["rmse_mv"]))
    ds = data["test_id"]
    worst = {"current": ds.current[wk], "v_true": ds.voltage[wk], "mean": preds["test_id"][0][wk],
             "std": preds["test_id"][1][wk],
             "title": f"worst in-range test trace ({ds.family[wk]}), "
                      f"RMSE {per_split['test_id']['rmse_mv'][wk]:.1f} mV"}
    plots.plot_failure_modes(figdir / "failure_modes.png", by_family, err_t, dt, worst)


def _fmt(x: Any, nd: int = 3) -> str:
    if isinstance(x, float):
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.{nd}f}"
    return str(x)


def render_report(m: dict[str, Any]) -> str:
    """Markdown tables summarising ``metrics.json`` (pasted into docs/RESULTS.md)."""
    lines: list[str] = [f"# Results tables ({m['config']['name']} config)", ""]
    sel = m["selection"]
    lines += ["## Architecture selection (validation split, single model each)", "",
              "| model | params | train time (s) | mean Gamma | exact spike count | "
              "median RMSE (mV) | median sub-threshold RMSE (mV) | speed-up vs scipy |",
              "|---|---|---|---|---|---|---|---|"]
    for k, s in sel["validation"].items():
        lines.append(f"| {KIND_NAME[k]} | {sel['n_parameters'][k]:,} | "
                     f"{_fmt(sel['train_seconds'][k], 0)} | {_fmt(s['gamma_mean'])} | "
                     f"{_fmt(s['spike_count_exact'])} | {_fmt(s['rmse_mv_median'], 2)} | "
                     f"{_fmt(s['sub_rmse_mv_median'], 2)} | "
                     f"{_fmt(sel['speedup_vs_scipy'][k], 0)}x |")
    lines += ["", f"Rule: {sel['rule']}. Selected: **{KIND_NAME[sel['winner']]}**.", ""]

    lines += ["## Held-out accuracy of the twin (ensemble mean)", "",
              "| split | traces | spiking | mean Gamma | median Gamma | exact spike count | "
              "spike-count R^2 | median RMSE (mV) | median sub-thr. RMSE (mV) | "
              "1st-spike latency err (ms) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for split in ("test_id", "test_ood_param", "test_ood_stim"):
        e = m["splits"][split]["ensemble"]
        lines.append(
            f"| {split} | {int(e['n_traces'])} | {_fmt(e['frac_spiking'], 2)} | "
            f"{_fmt(e['gamma_mean'])} | {_fmt(e['gamma_median'])} | "
            f"{_fmt(e['spike_count_exact'])} | {_fmt(e['spike_count_r2'])} | "
            f"{_fmt(e['rmse_mv_median'], 2)} | {_fmt(e['sub_rmse_mv_median'], 2)} | "
            f"{_fmt(e['latency_err_ms_median'], 2)} |")
    lines += ["", "Single ensemble member (average over members), same splits:", "",
              "| split | mean Gamma | exact spike count | median RMSE (mV) |", "|---|---|---|---|"]
    for split in ("test_id", "test_ood_param", "test_ood_stim"):
        e = m["splits"][split]["single_member_mean"]
        lines.append(f"| {split} | {_fmt(e['gamma_mean'])} | {_fmt(e['spike_count_exact'])} | "
                     f"{_fmt(e['rmse_mv_median'], 2)} |")
    lines += ["", "Summary-feature errors (mean absolute error, ensemble mean trace):", "",
              "| split | firing rate (Hz) | first-spike latency (ms) | mean sub-thr. V (mV) |"
              " peak V (mV) |", "|---|---|---|---|---|"]
    for split in ("test_id", "test_ood_param", "test_ood_stim"):
        e = m["splits"][split]["summary_feature_errors"]
        lines.append(f"| {split} | {_fmt(e['rate_hz_mae'], 2)} | "
                     f"{_fmt(e['first_latency_ms_mae'], 2)} | {_fmt(e['v_mean_sub_mv_mae'], 2)} | "
                     f"{_fmt(e['v_max_mv_mae'], 2)} |")
    lines += ["", "Mean Gamma by stimulus family (* = never seen in training):", "",
              "| " + " | ".join(m["family_gamma"]) + " |",
              "|" + "---|" * len(m["family_gamma"]),
              "| " + " | ".join(_fmt(v) for v in m["family_gamma"].values()) + " |", ""]

    u = m["uncertainty"]
    lines += ["## Uncertainty", "",
              "| split | median sigma (mV) | 2-sigma coverage | Spearman(sigma, RMSE) | "
              "flagged as extrapolation |", "|---|---|---|---|---|"]
    for split in ("test_id", "test_ood_param", "test_ood_stim"):
        s = m["splits"][split]
        lines.append(f"| {split} | {_fmt(s['uncertainty_mv_median'], 2)} | "
                     f"{_fmt(s['coverage_2sigma'])} | {_fmt(s['spearman_uncertainty_vs_rmse'])} | "
                     f"{_fmt(u['flag_rate'][split])} |")
    lines += ["",
              "- flag rate when sigma is combined with a parameter range check: "
              + ", ".join(f"{k} {_fmt(v)}" for k, v in u["flag_rate_with_range_check"].items()),
              "- flag threshold (95th pct of validation sigma): "
              f"{_fmt(u['flag_threshold_mv'], 2)} mV",
              f"- AUROC ID vs extrapolated params: {_fmt(u['auroc_id_vs_ood_param'])}; "
              f"ID vs novel stimuli: {_fmt(u['auroc_id_vs_ood_stim'])}",
              "- Spearman(sigma, RMSE), pooled test splits: "
              f"{_fmt(u['spearman_u_vs_rmse_pooled'])}",
              f"- Spearman(sigma, distance outside training box): "
              f"{_fmt(u['spearman_u_vs_extrapolation_distance'])}",
              f"- mean Gamma flagged / unflagged: {_fmt(u['gamma_mean_flagged'])} / "
              f"{_fmt(u['gamma_mean_unflagged'])}; median RMSE flagged / unflagged: "
              f"{_fmt(u['rmse_median_flagged'], 2)} / {_fmt(u['rmse_median_unflagged'], 2)} mV",
              ""]

    lines += ["## Speed (seconds per 100 ms trace, CPU)", "",
              "| method | batch | threads | ms / trace | speed-up vs scipy | "
              "speed-up vs numpy RK4 |",
              "|---|---|---|---|---|---|"]
    for k, s in m["speed"].items():
        lines.append(f"| {k} | {int(s['batch'])} | {int(s['threads'])} | "
                     f"{1e3 * s['sec_per_trace']:.4g} | {s['speedup_vs_scipy']:,.1f} | "
                     f"{s['speedup_vs_numpy_rk4']:,.2f} |")
    a = m["reference_agreement"]
    substeps = m["config"]["sim"]["rk4_substeps"]
    lines += ["", f"Reference check: dataset integrator (RK4, dt/{substeps}) "
              f"vs scipy LSODA (rtol 1e-9) on {int(a['n_traces'])} test traces: "
              f"max |dV| = {a['max_abs_err_mv']:.2e} mV, RMSE = {a['rmse_mv']:.2e} mV, "
              f"mean Gamma = {a['gamma_mean']:.3f}.", "",
              f"f-I curves: mean absolute rate error {m['fi']['mae_hz']} Hz.", "",
              f"Total pipeline time: {m['total_seconds']:.0f} s.", ""]
    return "\n".join(lines)


def tiny_config_dict() -> dict[str, Any]:
    """A seconds-scale configuration used by the end-to-end test."""
    return {
        "name": "tiny", "seed": 0,
        "sim": {"t_max_ms": 20.0},
        "data": {"n_train": 24, "n_val": 8, "n_test": 8, "n_workers": 1},
        "gru": {"hidden": 16, "epochs": 2, "batch_size": 8},
        "node": {"width": 16, "epochs": 2, "iters_per_epoch": 3, "windows": [4, 8],
                 "batch_size": 16},
        "ensemble": {"members": 2},
        "eval": {"speed_n_scipy": 2, "speed_batch_sizes": [1, 8], "n_scipy_check": 2,
                 "fi_currents": [0.0, 10.0], "min_speedup": 0.0},
    }


__all__ = [
    "PARAM_NAMES",
    "load_twin",
    "render_report",
    "run",
    "save_twin",
    "select_model",
    "tiny_config_dict",
]
