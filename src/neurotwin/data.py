"""Dataset generation: (parameters, stimulus) -> (voltage trace, summary)."""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .features import summary_table
from .hh import HHParams, resting_state, simulate_batch
from .stimuli import make_stimulus, sample_families

FloatArray = NDArray[np.float64]

#: Dataset splits and how they are sampled: (parameter box, OOD stimuli?).
SPLITS: dict[str, tuple[str, bool]] = {
    "train": ("train", False),
    "val": ("train", False),
    "test_id": ("train", False),
    "test_ood_param": ("ood", False),
    "test_ood_stim": ("train", True),
}


@dataclass
class Dataset:
    """A batch of simulated neurons.

    Attributes:
        theta: ``(B, 4)`` parameters ordered like ``hh.PARAM_NAMES``.
        current: ``(B, N)`` injected current (zero-order hold).
        states: ``(B, N, 4)`` true ``(V, m, h, n)`` at ``t = dt..N dt``.
        x0: ``(B, 4)`` resting state at ``t = 0``.
        family: ``(B,)`` stimulus family labels.
        dt: sampling interval (ms).
        n_fallback: traces that needed the stiff LSODA fallback.
    """

    theta: FloatArray
    current: FloatArray
    states: FloatArray
    x0: FloatArray
    family: NDArray[np.str_]
    dt: float
    n_fallback: int = 0

    def __len__(self) -> int:
        return int(self.theta.shape[0])

    @property
    def voltage(self) -> FloatArray:
        return self.states[:, :, 0]

    def subset(self, idx: NDArray[np.int64] | slice) -> Dataset:
        return Dataset(
            self.theta[idx],
            self.current[idx],
            self.states[idx],
            self.x0[idx],
            self.family[idx],
            self.dt,
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            theta=self.theta,
            current=self.current,
            states=self.states.astype(np.float32),
            x0=self.x0,
            family=self.family,
            dt=np.array(self.dt),
            n_fallback=np.array(self.n_fallback),
        )

    @staticmethod
    def load(path: str | Path) -> Dataset:
        with np.load(path) as z:
            return Dataset(
                theta=z["theta"].astype(np.float64),
                current=z["current"].astype(np.float64),
                states=z["states"].astype(np.float64),
                x0=z["x0"].astype(np.float64),
                family=z["family"].astype(np.str_),
                dt=float(z["dt"]),
                n_fallback=int(z["n_fallback"]) if "n_fallback" in z else 0,
            )


def sample_params(
    rng: np.random.Generator, n: int, lo: list[float], hi: list[float],
    exclude: tuple[list[float], list[float]] | None = None,
) -> FloatArray:
    """Uniform samples in ``[lo, hi]``; optionally reject points inside ``exclude``."""
    lo_a, hi_a = np.asarray(lo), np.asarray(hi)
    out: list[FloatArray] = []
    n_have = 0
    while n_have < n:
        cand = rng.uniform(lo_a, hi_a, size=(2 * n, len(lo)))
        if exclude is not None:
            ex_lo, ex_hi = np.asarray(exclude[0]), np.asarray(exclude[1])
            inside = np.all((cand >= ex_lo) & (cand <= ex_hi), axis=1)
            cand = cand[~inside]
        out.append(cand)
        n_have += cand.shape[0]
    return np.concatenate(out)[:n]


def _simulate_chunk(
    args: tuple[FloatArray, FloatArray, float, int, float],
) -> tuple[FloatArray, int]:
    theta, current, dt, substeps, temperature = args
    return simulate_batch(theta, current, dt, substeps, HHParams(temperature=temperature))


def simulate_many(
    theta: FloatArray,
    current: FloatArray,
    cfg: Config,
    n_workers: int | None = None,
    chunk: int = 128,
) -> tuple[FloatArray, int]:
    """Simulate many neurons with the batch integrator, in parallel chunks.

    Returns the states ``(B, N, 4)`` and the number of stiff-solver fallbacks.
    """
    n_workers = cfg.data.n_workers if n_workers is None else n_workers
    jobs = [
        (theta[i : i + chunk], current[i : i + chunk], cfg.sim.dt_ms, cfg.sim.rk4_substeps,
         cfg.sim.temperature)
        for i in range(0, theta.shape[0], chunk)
    ]
    if n_workers <= 1 or len(jobs) == 1:
        parts = [_simulate_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            parts = list(pool.map(_simulate_chunk, jobs))
    return np.concatenate([p[0] for p in parts], axis=0), sum(p[1] for p in parts)


def make_split(cfg: Config, split: str, n: int, seed: int) -> Dataset:
    """Sample parameters and stimuli for ``split`` and simulate them."""
    box, ood_stim = SPLITS[split]
    rng = np.random.default_rng(seed)
    train_lo, train_hi = cfg.params.bounds("train")
    if box == "train":
        theta = sample_params(rng, n, train_lo, train_hi)
    else:
        ood_lo, ood_hi = cfg.params.bounds("ood")
        theta = sample_params(rng, n, ood_lo, ood_hi, exclude=(train_lo, train_hi))
    families = sample_families(rng, n, ood_stim, cfg.stimulus)
    n_steps = cfg.sim.n_steps
    current = np.stack(
        [make_stimulus(f, rng, n_steps, cfg.sim.dt_ms, cfg.stimulus) for f in families]
    )
    states, n_fallback = simulate_many(theta, current, cfg)
    x0 = resting_state(theta, HHParams(temperature=cfg.sim.temperature))
    return Dataset(theta, current, states, x0, np.array(families, dtype=np.str_), cfg.sim.dt_ms,
                   n_fallback)


def split_sizes(cfg: Config) -> dict[str, int]:
    return {
        "train": cfg.data.n_train,
        "val": cfg.data.n_val,
        "test_id": cfg.data.n_test,
        "test_ood_param": cfg.data.n_test,
        "test_ood_stim": cfg.data.n_test,
    }


def build_datasets(cfg: Config, cache_dir: Path | None = None) -> dict[str, Dataset]:
    """Generate (or load from ``cache_dir``) every split defined in ``SPLITS``."""
    out: dict[str, Dataset] = {}
    for k, (split, n) in enumerate(split_sizes(cfg).items()):
        # The cache key covers everything that determines the samples.
        key = repr((cfg.seed, k, n, cfg.sim, cfg.params, cfg.stimulus))
        tag = hashlib.sha1(key.encode()).hexdigest()[:10]
        path = cache_dir / f"{split}-{tag}.npz" if cache_dir is not None else None
        if path is not None and path.exists():
            out[split] = Dataset.load(path)
            continue
        ds = make_split(cfg, split, n, seed=cfg.seed * 1000 + k)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            ds.save(path)
        out[split] = ds
    return out


def dataset_summaries(ds: Dataset) -> dict[str, FloatArray]:
    """Electrophysiological summary features of every trace in ``ds``."""
    return summary_table(ds.voltage, ds.dt)
