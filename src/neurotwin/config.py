"""Experiment configuration (YAML -> typed dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .hh import PARAM_NAMES
from .stimuli import StimulusConfig

Range = tuple[float, float]


@dataclass(frozen=True)
class SimConfig:
    t_max_ms: float = 100.0
    dt_ms: float = 0.1
    rk4_substeps: int = 5
    temperature: float = 6.3
    rtol: float = 1e-6
    atol: float = 1e-8

    @property
    def n_steps(self) -> int:
        return round(self.t_max_ms / self.dt_ms)


@dataclass(frozen=True)
class ParamConfig:
    """Training box (in-distribution) and the wider extrapolation box."""

    train_box: dict[str, Range] = field(
        default_factory=lambda: {
            "g_na": (100.0, 140.0),
            "g_k": (28.0, 44.0),
            "g_l": (0.2, 0.4),
            "c_m": (0.85, 1.15),
        }
    )
    ood_box: dict[str, Range] = field(
        default_factory=lambda: {
            "g_na": (70.0, 170.0),
            "g_k": (18.0, 54.0),
            "g_l": (0.1, 0.6),
            "c_m": (0.7, 1.3),
        }
    )

    def bounds(self, which: str) -> tuple[list[float], list[float]]:
        box = self.train_box if which == "train" else self.ood_box
        lo = [float(box[n][0]) for n in PARAM_NAMES]
        hi = [float(box[n][1]) for n in PARAM_NAMES]
        return lo, hi


@dataclass(frozen=True)
class DataConfig:
    n_train: int = 600
    n_val: int = 100
    n_test: int = 100  # per test split
    n_workers: int = 2


@dataclass(frozen=True)
class GRUConfig:
    hidden: int = 64
    layers: int = 2
    patch: int = 5
    epochs: int = 15
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 1e-5


@dataclass(frozen=True)
class NODEConfig:
    width: int = 64
    depth: int = 3
    substeps: int = 2
    epochs: int = 10
    iters_per_epoch: int = 50
    windows: tuple[int, ...] = (8, 32)
    batch_size: int = 256
    lr: float = 2e-3


@dataclass(frozen=True)
class EnsembleConfig:
    members: int = 3


@dataclass(frozen=True)
class EvalConfig:
    coincidence_window_ms: float = 2.0
    # Architecture selection: the most accurate candidate whose batched inference
    # is at least this many times faster than scipy LSODA (the twin's purpose).
    min_speedup: float = 100.0
    # Extra batch sizes timed on a CUDA device when one is available.
    gpu_batch_sizes: tuple[int, ...] = (1, 1024, 8192)
    flag_quantile: float = 0.95
    speed_n_scipy: int = 20
    speed_batch_sizes: tuple[int, ...] = (1, 256)
    n_scipy_check: int = 20
    fi_currents: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 45.0)


@dataclass(frozen=True)
class Config:
    name: str = "small"
    seed: int = 0
    torch_threads: int = 0  # 0 = torch default
    parallel_jobs: int = 1  # independent models trained side by side (processes)
    device: str = "auto"  # "auto" (CUDA if available), "cpu", "cuda", "cuda:N"
    sim: SimConfig = field(default_factory=SimConfig)
    params: ParamConfig = field(default_factory=ParamConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    node: NODEConfig = field(default_factory=NODEConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _coerce(value: Any, default: Any) -> Any:
    """Convert YAML lists to tuples where the default is a tuple."""
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(default, dict) and isinstance(value, dict):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in value.items()}
    return value


def _build(cls: type[Any], values: dict[str, Any] | None) -> Any:
    default = cls()
    if not values:
        return default
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {k: _coerce(v, getattr(default, k)) for k, v in values.items()}
    return cls(**kwargs)


def config_from_dict(d: dict[str, Any]) -> Config:
    sections: dict[str, type[Any]] = {
        "sim": SimConfig,
        "params": ParamConfig,
        "stimulus": StimulusConfig,
        "data": DataConfig,
        "gru": GRUConfig,
        "node": NODEConfig,
        "ensemble": EnsembleConfig,
        "eval": EvalConfig,
    }
    scalars = ("name", "seed", "torch_threads", "parallel_jobs", "device")
    unknown = set(d) - set(sections) - set(scalars)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    kwargs: dict[str, Any] = {k: _build(cls, d.get(k)) for k, cls in sections.items()}
    for key in scalars:
        if key in d:
            kwargs[key] = d[key]
    cfg = Config(**kwargs)
    for box in (cfg.params.train_box, cfg.params.ood_box):
        if set(box) != set(PARAM_NAMES):
            raise ValueError(f"parameter boxes must define exactly {PARAM_NAMES}")
    if not (cfg.device in ("auto", "cpu", "cuda") or cfg.device.startswith("cuda:")):
        raise ValueError(f"device must be auto, cpu, cuda or cuda:N, not {cfg.device!r}")
    if cfg.sim.n_steps % cfg.gru.patch != 0:
        raise ValueError("t_max_ms / dt_ms must be a multiple of gru.patch")
    return cfg


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return config_from_dict(raw)
