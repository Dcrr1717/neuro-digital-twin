"""The trained twin committed in results/full/twin must reproduce the simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurotwin.features import coincidence_factor, spike_times
from neurotwin.hh import simulate_batch
from neurotwin.pipeline import load_twin

TWIN = Path(__file__).resolve().parents[1] / "results" / "full" / "twin"


@pytest.mark.skipif(not (TWIN / "normalizer.json").exists(), reason="no shipped twin")
def test_shipped_twin_matches_simulator() -> None:
    twin = load_twin(TWIN)
    assert len(twin.members) == 5
    dt, n = 0.1, 1000
    theta = np.array([[120.0, 36.0, 0.3, 1.0], [110.0, 40.0, 0.25, 0.9]])
    current = np.zeros((2, n))
    current[:, 100:800] = 10.0  # 70 ms step of 10 uA/cm^2
    truth = simulate_batch(theta, current, dt)[0][:, :, 0]
    mean, std, members = twin.predict(theta, current)
    for k in range(2):
        t_true = spike_times(truth[k], dt)
        for m in members[:, k]:
            gamma = coincidence_factor(t_true, spike_times(m, dt), 2.0, n * dt)
            assert gamma > 0.8
        assert np.median(np.abs(mean[k] - truth[k])) < 2.0  # mV
    assert np.all(std.mean(axis=1) < 1.23)  # not flagged as extrapolation
