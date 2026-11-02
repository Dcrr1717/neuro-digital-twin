"""Tests of the reference Hodgkin-Huxley simulator."""

from __future__ import annotations

import numpy as np
import pytest

from neurotwin.features import spike_times
from neurotwin.hh import (
    HHParams,
    derivatives,
    rates,
    resting_state,
    simulate_rk4_batch,
    simulate_scipy,
)

CLASSIC = np.array([120.0, 36.0, 0.3, 1.0])
DT = 0.1


def step(n: int, amp: float, start: int = 50, stop: int | None = None) -> np.ndarray:
    i = np.zeros(n)
    i[start:stop] = amp
    return i


def test_rates_continuous_at_removable_singularities() -> None:
    v = np.array([-40.0 - 1e-5, -40.0, -40.0 + 1e-5, -55.0 - 1e-5, -55.0, -55.0 + 1e-5])
    a_m, _, _, _, a_n, _ = rates(v)
    assert np.all(np.isfinite(a_m)) and np.all(np.isfinite(a_n))
    assert a_m[1] == pytest.approx(1.0, rel=1e-6)  # 0.1 * 10
    assert a_n[4] == pytest.approx(0.1, rel=1e-6)  # 0.01 * 10
    assert np.ptp(a_m[:3]) < 1e-5 and np.ptp(a_n[3:]) < 1e-6


def test_resting_state_is_an_equilibrium() -> None:
    theta = np.array([CLASSIC, [100.0, 28.0, 0.2, 0.85], [140.0, 44.0, 0.4, 1.15]])
    x0 = resting_state(theta)
    assert x0[0, 0] == pytest.approx(-65.0, abs=0.1)  # classic HH rests near -65 mV
    d = derivatives(x0, np.zeros(3), theta)
    assert np.max(np.abs(d)) < 1e-8


def test_q10_temperature_factor() -> None:
    assert HHParams().phi == pytest.approx(1.0)
    assert HHParams(temperature=16.3).phi == pytest.approx(3.0)


def test_with_vector_roundtrip() -> None:
    p = HHParams().with_vector(np.array([110.0, 30.0, 0.25, 0.9]))
    assert p.g_na == 110.0 and p.c_m == 0.9 and p.e_k == -77.0
    np.testing.assert_allclose(p.vector(), [110.0, 30.0, 0.25, 0.9])


def test_no_input_stays_at_rest() -> None:
    v = simulate_rk4_batch(CLASSIC[None], np.zeros((1, 300)), DT)[0, :, 0]
    assert np.max(np.abs(v - v[0])) < 1e-6


def test_suprathreshold_step_fires_repetitively_and_subthreshold_does_not() -> None:
    n = 1000
    cur = np.stack([step(n, 1.0), step(n, 10.0)])
    v = simulate_rk4_batch(np.stack([CLASSIC, CLASSIC]), cur, DT)[:, :, 0]
    assert len(spike_times(v[0], DT)) == 0
    spikes = spike_times(v[1], DT)
    assert 4 <= len(spikes) <= 8  # ~60-70 Hz at 10 uA/cm^2, 6.3 degC
    assert spikes[0] > 5.0  # nothing before the step onset
    assert 30.0 < np.max(v[1]) < 50.0  # overshoot below E_Na = 50 mV


def test_anode_break_excitation() -> None:
    """Release from hyperpolarisation triggers a rebound spike (HH 1952)."""
    v = simulate_rk4_batch(CLASSIC[None], step(600, -15.0, 50, 350)[None], DT)[0, :, 0]
    spikes = spike_times(v, DT)
    assert len(spikes) == 1 and spikes[0] > 35.0


def test_rk4_matches_scipy_lsoda() -> None:
    rng = np.random.default_rng(0)
    n = 600
    cur = np.stack([step(n, 12.0), 4.0 + 3.0 * rng.standard_normal(n)])
    theta = np.stack([CLASSIC, [105.0, 40.0, 0.25, 1.1]])
    fast = simulate_rk4_batch(theta, cur, DT, substeps=5)
    for k in range(2):
        ref = simulate_scipy(theta[k], cur[k], DT, rtol=1e-9, atol=1e-11)
        assert ref.shape == (n, 4)
        assert np.max(np.abs(ref[:, 0] - fast[k, :, 0])) < 0.05  # mV
        np.testing.assert_allclose(spike_times(ref[:, 0], DT), spike_times(fast[k, :, 0], DT),
                                   atol=1e-3)


def test_gating_variables_stay_in_unit_interval() -> None:
    cur = step(800, 30.0)[None]
    s = simulate_rk4_batch(CLASSIC[None], cur, DT)[0]
    assert np.all((s[:, 1:] >= 0.0) & (s[:, 1:] <= 1.0))
