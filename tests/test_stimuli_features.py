"""Tests of stimulus generators and spike-train features."""

from __future__ import annotations

import numpy as np
import pytest

from neurotwin.features import coincidence_factor, spike_mask, spike_times, summary_features
from neurotwin.stimuli import ALL_FAMILIES, StimulusConfig, make_stimulus, sample_families


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_stimulus_shape_onset_and_determinism(family: str) -> None:
    cfg = StimulusConfig(onset_ms=(10.0, 10.0))
    a = make_stimulus(family, np.random.default_rng(3), 1000, 0.1, cfg)
    b = make_stimulus(family, np.random.default_rng(3), 1000, 0.1, cfg)
    assert a.shape == (1000,)
    np.testing.assert_array_equal(a, b)
    assert np.all(a[:100] == 0.0)  # silent before onset
    assert np.all(np.isfinite(a))


def test_step_amplitude_range() -> None:
    cfg = StimulusConfig()
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = make_stimulus("step", rng, 1000, 0.1, cfg)
        nz = a[a != 0]
        assert nz.size == 0 or cfg.step_amp[0] <= nz[0] <= cfg.step_amp[1]
        s = make_stimulus("strong_step", rng, 1000, 0.1, cfg)
        assert np.max(s) >= cfg.strong_step_amp[0]


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError):
        make_stimulus("nope", np.random.default_rng(0), 10, 0.1)


def test_sample_families_respects_ood_flag() -> None:
    rng = np.random.default_rng(0)
    assert set(sample_families(rng, 200, ood=True)) <= {"strong_step", "chirp"}
    assert set(sample_families(rng, 200, ood=False)) == {"step", "ramp", "noise", "pulses"}


def test_spike_times_interpolates_and_rearms() -> None:
    dt = 0.1
    v = np.full(100, -65.0)
    v[10], v[11] = -10.0, 10.0  # crossing halfway between samples 10 and 11
    v[12] = 5.0
    v[13] = 2.0  # dips but stays above re-arm level: same spike
    v[14] = -5.0
    v[15] = 3.0  # would be a second crossing without re-arming
    t = spike_times(v, dt)
    assert len(t) == 1
    assert t[0] == pytest.approx(dt * 11.5)
    v[20:30] = -70.0
    v[40] = 20.0
    assert len(spike_times(v, dt)) == 2


def test_coincidence_factor_limits() -> None:
    t = np.array([10.0, 30.0, 50.0, 70.0])
    assert coincidence_factor(t, t, 2.0, 100.0) == pytest.approx(1.0)
    assert coincidence_factor(t, t + 1.0, 2.0, 100.0) == pytest.approx(1.0)
    assert coincidence_factor(t, t + 10.0, 2.0, 100.0) < 0.0
    assert coincidence_factor(np.empty(0), np.empty(0), 2.0, 100.0) == 1.0
    assert coincidence_factor(t, np.empty(0), 2.0, 100.0) == 0.0
    assert coincidence_factor(np.empty(0), t, 2.0, 100.0) == 0.0
    # one model spike cannot be matched to two data spikes
    assert coincidence_factor(np.array([10.0, 11.0]), np.array([10.5]), 2.0, 100.0) < 1.0


def test_summary_features() -> None:
    dt = 0.1
    v = np.full(1000, -65.0)
    for k in (100, 300, 500):
        v[k : k + 5] = 30.0
    f = summary_features(v, dt)
    assert f["n_spikes"] == 3
    assert f["rate_hz"] == pytest.approx(30.0)
    assert f["first_latency_ms"] == pytest.approx(10.0, abs=0.1)
    assert f["isi_cv"] == pytest.approx(0.0, abs=1e-6)
    assert f["v_mean_sub_mv"] == pytest.approx(-65.0)
    mask = spike_mask(1000, dt, np.array([10.0]))
    assert mask.sum() == pytest.approx(50, abs=1)
