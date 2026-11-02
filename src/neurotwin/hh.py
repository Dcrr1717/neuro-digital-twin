"""Reference Hodgkin-Huxley simulator (the "ground truth" the twin emulates).

Single-compartment Hodgkin & Huxley (1952) squid-axon model in the modern
sign convention (resting potential near -65 mV), with Q10 temperature scaling
of the gating kinetics.

Units: time in ms, voltage in mV, current density in uA/cm^2, conductance
density in mS/cm^2, capacitance in uF/cm^2.

Two integrators are provided:

* :func:`simulate_scipy` -- adaptive LSODA (scipy/ODEPACK), one neuron at a
  time. This is how the model is conventionally simulated and it is the
  baseline for the speed-up numbers.
* :func:`simulate_rk4_batch` -- fixed-step classical RK4 (numpy, vectorised
  over many neurons) with a 0.02 ms internal step. It agrees with LSODA to
  ~1e-2 mV (see ``tests/test_hh.py`` and ``docs/RESULTS.md``), i.e. it is a
  converged solution of the same equations. Wrapped by :func:`simulate_batch`
  (which re-solves the rare stiff, diverging traces with LSODA) it generates
  the training and test datasets because it is much cheaper per trace.

Stimulus convention (zero-order hold): ``current[k]`` is the injected current
on the interval ``[k*dt, (k+1)*dt)`` and the returned ``V[k]`` is the membrane
potential at the *end* of that interval, ``t = (k+1)*dt``. The initial state at
``t = 0`` is the resting equilibrium of the given parameters (no input).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import odeint

FloatArray = NDArray[np.float64]

#: Names of the parameters the digital twin is conditioned on.
PARAM_NAMES: Final[tuple[str, ...]] = ("g_na", "g_k", "g_l", "c_m")

#: State variables of the model, in order.
STATE_NAMES: Final[tuple[str, ...]] = ("V", "m", "h", "n")


@dataclass(frozen=True)
class HHParams:
    """Biophysical parameters of the single-compartment HH model."""

    g_na: float = 120.0  # maximal Na+ conductance, mS/cm^2
    g_k: float = 36.0  # maximal K+ conductance, mS/cm^2
    g_l: float = 0.3  # leak conductance, mS/cm^2
    c_m: float = 1.0  # membrane capacitance, uF/cm^2
    e_na: float = 50.0  # Na+ reversal, mV
    e_k: float = -77.0  # K+ reversal, mV
    e_l: float = -54.387  # leak reversal, mV
    temperature: float = 6.3  # degrees Celsius

    @property
    def phi(self) -> float:
        """Q10 = 3 temperature factor relative to 6.3 degC (HH 1952)."""
        return float(3.0 ** ((self.temperature - 6.3) / 10.0))

    def with_vector(self, theta: FloatArray) -> HHParams:
        """Return a copy with the varied parameters (``PARAM_NAMES``) replaced."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        for name, value in zip(PARAM_NAMES, np.asarray(theta, dtype=np.float64), strict=True):
            values[name] = float(value)
        return HHParams(**values)

    def vector(self) -> FloatArray:
        """The varied parameters as a vector ordered like ``PARAM_NAMES``."""
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# Rate functions (vectorised; valid for scalars and arrays)
# ---------------------------------------------------------------------------


def _vtrap(x: FloatArray, y: float) -> FloatArray:
    """Compute ``x / (exp(x / y) - 1)`` with the removable singularity at 0."""
    z = x / y
    small = np.abs(z) < 1e-6
    safe = np.where(small, 1.0, z)
    return np.where(small, y * (1.0 - z / 2.0), x / np.expm1(safe))


Rates = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]


def rates(v: FloatArray) -> Rates:
    """Opening/closing rates (1/ms) at 6.3 degC: (a_m, b_m, a_h, b_h, a_n, b_n)."""
    v = np.asarray(v, dtype=np.float64)
    a_m = 0.1 * _vtrap(-(v + 40.0), 10.0)
    b_m = 4.0 * np.exp(-(v + 65.0) / 18.0)
    a_h = 0.07 * np.exp(-(v + 65.0) / 20.0)
    b_h = 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    a_n = 0.01 * _vtrap(-(v + 55.0), 10.0)
    b_n = 0.125 * np.exp(-(v + 65.0) / 80.0)
    return a_m, b_m, a_h, b_h, a_n, b_n


def steady_state_gates(v: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Steady-state activation/inactivation ``(m_inf, h_inf, n_inf)``."""
    a_m, b_m, a_h, b_h, a_n, b_n = rates(v)
    return a_m / (a_m + b_m), a_h / (a_h + b_h), a_n / (a_n + b_n)


def _param_columns(theta: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    return theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]


def resting_state(theta: FloatArray, base: HHParams | None = None) -> FloatArray:
    """Resting equilibrium ``(V, m, h, n)`` for each row of ``theta`` (no input).

    Solves ``I_ion(V) = 0`` with the gates at steady state by vectorised
    bisection on ``[-90, -40]`` mV. Returns an array of shape ``(B, 4)``.
    """
    base = base or HHParams()
    g_na, g_k, g_l, _ = _param_columns(theta)

    def i_ion(v: FloatArray) -> FloatArray:
        m, h, n = steady_state_gates(v)
        return (
            g_na * m**3 * h * (v - base.e_na)
            + g_k * n**4 * (v - base.e_k)
            + g_l * (v - base.e_l)
        )

    lo = np.full_like(g_na, -90.0)
    hi = np.full_like(g_na, -40.0)
    # i_ion(lo) < 0 (inward leak/K driving force dominates) and i_ion(hi) > 0
    # for every physiological parameter set considered in this project.
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        positive = i_ion(mid) > 0.0
        hi = np.where(positive, mid, hi)
        lo = np.where(positive, lo, mid)
    v_rest = 0.5 * (lo + hi)
    m, h, n = steady_state_gates(v_rest)
    return np.stack([v_rest, m, h, n], axis=1)


def derivatives(
    state: FloatArray, current: FloatArray, theta: FloatArray, base: HHParams | None = None
) -> FloatArray:
    """Vectorised HH vector field. ``state`` is ``(B, 4)``, ``current`` is ``(B,)``."""
    base = base or HHParams()
    g_na, g_k, g_l, c_m = _param_columns(theta)
    v, m, h, n = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    a_m, b_m, a_h, b_h, a_n, b_n = rates(v)
    phi = base.phi
    i_ion = (
        g_na * m**3 * h * (v - base.e_na)
        + g_k * n**4 * (v - base.e_k)
        + g_l * (v - base.e_l)
    )
    dv = (current - i_ion) / c_m
    dm = phi * (a_m * (1.0 - m) - b_m * m)
    dh = phi * (a_h * (1.0 - h) - b_h * h)
    dn = phi * (a_n * (1.0 - n) - b_n * n)
    return np.stack([dv, dm, dh, dn], axis=1)


# ---------------------------------------------------------------------------
# Gold-standard integrator (scipy LSODA)
# ---------------------------------------------------------------------------


def _scalar_rhs_factory(
    current: FloatArray, dt: float, p: HHParams
) -> tuple[Callable[[float, list[float]], list[float]], int]:
    """Fast scalar right-hand side for ``solve_ivp`` (pure-Python math)."""
    import math

    i_list = current.tolist()
    n_steps = len(i_list)
    g_na, g_k, g_l, c_m = p.g_na, p.g_k, p.g_l, p.c_m
    e_na, e_k, e_l, phi = p.e_na, p.e_k, p.e_l, p.phi
    exp = math.exp

    def vtrap(x: float, y: float) -> float:
        z = x / y
        if abs(z) < 1e-6:
            return y * (1.0 - z / 2.0)
        if z > 700.0:  # x / (e^z - 1) underflows to 0 (only off-trajectory)
            return 0.0
        return x / math.expm1(z)

    inv_dt = 1.0 / dt

    def rhs(t: float, y: list[float]) -> list[float]:
        # The tiny offset makes t = k*dt (a segment start) map to segment k
        # despite floating-point round-off.
        k = int(t * inv_dt + 1e-7)
        if k >= n_steps:
            k = n_steps - 1
        elif k < 0:
            k = 0
        v, m, h, n = y[0], y[1], y[2], y[3]
        a_m = 0.1 * vtrap(-(v + 40.0), 10.0)
        b_m = 4.0 * exp(-(v + 65.0) / 18.0)
        a_h = 0.07 * exp(-(v + 65.0) / 20.0)
        b_h = 1.0 / (1.0 + exp(-(v + 35.0) / 10.0))
        a_n = 0.01 * vtrap(-(v + 55.0), 10.0)
        b_n = 0.125 * exp(-(v + 65.0) / 80.0)
        i_ion = (
            g_na * m * m * m * h * (v - e_na)
            + g_k * n * n * n * n * (v - e_k)
            + g_l * (v - e_l)
        )
        return [
            (i_list[k] - i_ion) / c_m,
            phi * (a_m * (1.0 - m) - b_m * m),
            phi * (a_h * (1.0 - h) - b_h * h),
            phi * (a_n * (1.0 - n) - b_n * n),
        ]

    return rhs, n_steps


def simulate_scipy(
    theta: FloatArray,
    current: FloatArray,
    dt: float,
    base: HHParams | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> FloatArray:
    """Integrate one neuron with adaptive LSODA (scipy/ODEPACK).

    This is the conventional way a modeller would simulate the neuron and is the
    reference against which both the fast batch integrator and the surrogate are
    checked, and the baseline for the speed-up numbers.

    Args:
        theta: varied parameters, shape ``(4,)`` ordered like ``PARAM_NAMES``.
        current: injected current, shape ``(N,)``, zero-order hold on ``dt``.
        dt: sampling interval of stimulus and output, ms.
        base: fixed parameters (reversal potentials, temperature).
        rtol, atol: LSODA tolerances.

    Returns:
        States at ``t = dt, 2 dt, ..., N dt`` with shape ``(N, 4)``.
    """
    base = base or HHParams()
    p = base.with_vector(theta)
    rhs, n_steps = _scalar_rhs_factory(np.asarray(current, dtype=np.float64), dt, p)
    y0 = resting_state(p.vector()[None, :], base)[0]
    t_grid = dt * np.arange(0, n_steps + 1)
    # ``tcrit`` stops the solver from stepping across the stimulus
    # discontinuities and ``hmax = dt`` guarantees that no zero-order-hold
    # segment is skipped.
    y = odeint(
        rhs,
        y0,
        t_grid,
        tfirst=True,
        rtol=rtol,
        atol=atol,
        hmax=dt,
        tcrit=t_grid,
        mxstep=50_000,
    )
    return np.asarray(y[1:], dtype=np.float64)


# ---------------------------------------------------------------------------
# Vectorised fixed-step baseline (numpy RK4)
# ---------------------------------------------------------------------------


def simulate_rk4_batch(
    theta: FloatArray,
    current: FloatArray,
    dt: float,
    substeps: int = 5,
    base: HHParams | None = None,
) -> FloatArray:
    """Integrate a batch of neurons with fixed-step RK4 (vectorised numpy).

    Args:
        theta: shape ``(B, 4)``.
        current: shape ``(B, N)``, zero-order hold on ``dt``.
        dt: output/stimulus interval, ms.
        substeps: RK4 steps per output interval (internal step ``dt/substeps``).

    Returns:
        States of shape ``(B, N, 4)`` at ``t = dt, ..., N dt``.
    """
    base = base or HHParams()
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    current = np.atleast_2d(np.asarray(current, dtype=np.float64))
    n_batch, n_steps = current.shape
    h = dt / substeps
    y = resting_state(theta, base)
    out = np.empty((n_batch, n_steps, 4), dtype=np.float64)
    for k in range(n_steps):
        i_k = current[:, k]
        for _ in range(substeps):
            k1 = derivatives(y, i_k, theta, base)
            k2 = derivatives(y + 0.5 * h * k1, i_k, theta, base)
            k3 = derivatives(y + 0.5 * h * k2, i_k, theta, base)
            k4 = derivatives(y + h * k3, i_k, theta, base)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[:, k, :] = y
    return out


def _diverged(states: FloatArray) -> NDArray[np.bool_]:
    """Traces whose fixed-step integration went unstable."""
    finite = np.isfinite(states).all(axis=(1, 2))
    v_ok = np.abs(np.nan_to_num(states[:, :, 0], nan=1e9)).max(axis=1) < 250.0
    gates = np.nan_to_num(states[:, :, 1:], nan=-1.0)
    g_ok = ((gates > -1e-3) & (gates < 1.0 + 1e-3)).all(axis=(1, 2))
    return np.asarray(~(finite & v_ok & g_ok))


def simulate_batch(
    theta: FloatArray,
    current: FloatArray,
    dt: float,
    substeps: int = 5,
    base: HHParams | None = None,
) -> tuple[FloatArray, int]:
    """Dataset integrator: vectorised RK4 with a stiff-solver safety net.

    Explicit RK4 becomes unstable when a gate time constant drops below
    ~``dt / (2.8 substeps)``; in HH this happens for the Na activation ``m`` at
    strongly hyperpolarised potentials (tau_m ~ 7 us at -130 mV). Such traces
    are detected and recomputed with LSODA, which is stiff-capable.

    Returns:
        ``(states, n_fallback)`` with ``states`` shaped ``(B, N, 4)``.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    current = np.atleast_2d(np.asarray(current, dtype=np.float64))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        states = simulate_rk4_batch(theta, current, dt, substeps, base)
    bad = np.flatnonzero(_diverged(states))
    for k in bad:
        states[k] = simulate_scipy(theta[k], current[k], dt, base, rtol=1e-8, atol=1e-10)
    return states, int(bad.size)
