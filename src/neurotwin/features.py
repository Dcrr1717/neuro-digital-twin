"""Electrophysiological features and spike-train similarity measures."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SPIKE_THRESHOLD_MV = 0.0
REARM_MV = -30.0


def spike_times(
    v: FloatArray, dt: float, threshold: float = SPIKE_THRESHOLD_MV, rearm: float = REARM_MV
) -> FloatArray:
    """Upward threshold crossings of one trace, in ms.

    ``v[k]`` is sampled at ``t = (k + 1) dt`` (see :mod:`neurotwin.hh`). Crossing
    times are linearly interpolated. After a spike the detector re-arms only
    once the trace falls below ``rearm``, so a jittery surrogate cannot produce
    several "spikes" on one action potential.
    """
    v = np.asarray(v, dtype=np.float64)
    up = np.flatnonzero((v[:-1] < threshold) & (v[1:] >= threshold)) + 1
    if up.size == 0:
        return np.empty(0)
    below = v < rearm
    times: list[float] = []
    last = -1
    for k in up:
        if last >= 0 and not below[last:k].any():
            continue
        frac = (threshold - v[k - 1]) / (v[k] - v[k - 1])
        times.append(dt * (k + frac))
        last = int(k)
    return np.asarray(times, dtype=np.float64)


def coincidence_factor(
    t_data: FloatArray, t_model: FloatArray, delta: float, duration: float
) -> float:
    """Coincidence factor Gamma (Kistler et al. 1997; Jolivet et al. 2008).

    Gamma = 1 for identical trains, ~0 for a Poisson train with the right
    rate, and is undefined (NaN) only when the model rate exceeds 1/(2 delta).
    By convention two silent trains have Gamma = 1.
    """
    n_d, n_m = len(t_data), len(t_model)
    if n_d == 0 and n_m == 0:
        return 1.0
    if n_d == 0 or n_m == 0:
        return 0.0
    # One-to-one greedy matching in time order.
    used = np.zeros(n_m, dtype=bool)
    n_coinc = 0
    for t in t_data:
        d = np.abs(t_model - t)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= delta:
            used[j] = True
            n_coinc += 1
    rate = n_m / duration
    norm = 1.0 - 2.0 * rate * delta
    if norm <= 0.0:
        return float("nan")
    expected = 2.0 * rate * delta * n_d
    return float((n_coinc - expected) / (0.5 * (n_d + n_m)) / norm)


def spike_mask(
    n_steps: int, dt: float, times: FloatArray, before: float = 1.0, after: float = 4.0
) -> BoolArray:
    """Boolean mask of samples within ``[t - before, t + after]`` of any spike."""
    t = dt * np.arange(1, n_steps + 1)
    mask = np.zeros(n_steps, dtype=bool)
    for s in times:
        mask |= (t >= s - before) & (t <= s + after)
    return mask


def summary_features(v: FloatArray, dt: float) -> dict[str, float]:
    """Standard summary statistics of one voltage trace."""
    n = len(v)
    duration = n * dt
    st = spike_times(v, dt)
    isi = np.diff(st)
    sub = ~spike_mask(n, dt, st)
    return {
        "n_spikes": float(len(st)),
        "rate_hz": 1000.0 * len(st) / duration,
        "first_latency_ms": float(st[0]) if len(st) else float("nan"),
        "isi_cv": float(isi.std() / isi.mean()) if len(isi) >= 2 else float("nan"),
        "v_mean_sub_mv": float(v[sub].mean()) if sub.any() else float("nan"),
        "v_max_mv": float(np.max(v)),
    }


def summary_table(v: FloatArray, dt: float) -> dict[str, FloatArray]:
    """:func:`summary_features` for every row of ``v`` (shape ``(B, N)``)."""
    rows = [summary_features(row, dt) for row in np.atleast_2d(v)]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}
