"""Tests of configuration loading and dataset generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurotwin.config import config_from_dict, load_config
from neurotwin.data import Dataset, build_datasets, make_split, sample_params
from neurotwin.pipeline import tiny_config_dict

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["small", "full"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.name == name
    assert cfg.sim.n_steps % cfg.gru.patch == 0
    lo, hi = cfg.params.bounds("train")
    olo, ohi = cfg.params.bounds("ood")
    assert all(a <= b for a, b in zip(olo, lo, strict=True))
    assert all(a >= b for a, b in zip(ohi, hi, strict=True))


def test_full_config_is_larger_than_small() -> None:
    small = load_config(ROOT / "configs" / "small.yaml")
    full = load_config(ROOT / "configs" / "full.yaml")
    assert full.data.n_train > small.data.n_train
    assert full.ensemble.members >= small.ensemble.members


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        config_from_dict({"gru": {"hiden": 3}})
    with pytest.raises(ValueError):
        config_from_dict({"oops": {}})
    with pytest.raises(ValueError):
        config_from_dict({"sim": {"t_max_ms": 10.3}})  # 103 steps not divisible by patch 5


def test_sample_params_excludes_training_box() -> None:
    rng = np.random.default_rng(0)
    lo, hi = [0.0, 0.0], [1.0, 1.0]
    x = sample_params(rng, 500, [-1.0, -1.0], [2.0, 2.0], exclude=(lo, hi))
    assert x.shape == (500, 2)
    inside = np.all((x >= 0.0) & (x <= 1.0), axis=1)
    assert not inside.any()


def test_splits_are_deterministic_and_well_formed(tmp_path: Path) -> None:
    cfg = config_from_dict(tiny_config_dict())
    a = make_split(cfg, "test_ood_param", 6, seed=5)
    b = make_split(cfg, "test_ood_param", 6, seed=5)
    np.testing.assert_array_equal(a.states, b.states)
    assert a.states.shape == (6, cfg.sim.n_steps, 4)
    lo, hi = cfg.params.bounds("train")
    assert not np.all((a.theta >= lo) & (a.theta <= hi), axis=1).any()
    assert set(make_split(cfg, "test_ood_stim", 6, seed=1).family) <= {"strong_step", "chirp"}

    a.save(tmp_path / "x.npz")
    c = Dataset.load(tmp_path / "x.npz")
    np.testing.assert_allclose(c.states, a.states, atol=1e-4)  # stored as float32
    assert c.dt == a.dt and list(c.family) == list(a.family)
    assert len(c.subset(slice(0, 2))) == 2


def test_build_datasets_caches(tmp_path: Path) -> None:
    cfg = config_from_dict(tiny_config_dict())
    d1 = build_datasets(cfg, tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 5
    d2 = build_datasets(cfg, tmp_path)
    np.testing.assert_allclose(d1["val"].states, d2["val"].states, atol=1e-4)
    assert len(d1["train"]) == cfg.data.n_train
