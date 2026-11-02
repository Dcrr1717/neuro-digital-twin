"""Stimulus protocols (injected current waveforms, uA/cm^2).

In-distribution families (used for training and the in-distribution tests):

* ``step``   -- current step (depolarising or hyperpolarising) of random
  amplitude, onset and duration; probes rheobase, repetitive firing and
  anode-break (rebound) excitation.
* ``ramp``   -- linearly rising current; probes accommodation and the
  threshold for repetitive firing.
* ``noise``  -- Ornstein-Uhlenbeck (coloured) noise around a mean; an in-vivo-
  like fluctuating input (Mainen & Sejnowski 1995; Destexhe et al. 2001).
* ``pulses`` -- train of brief, strong pulses at a random rate; probes
  refractoriness and entrainment.

Out-of-distribution families (never seen in training):

* ``strong_step`` -- steps 25-60 uA/cm^2, i.e. beyond the training amplitude.
* ``chirp``       -- sinusoid whose frequency sweeps 2-200 Hz (resonance).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Range = tuple[float, float]

ID_FAMILIES: tuple[str, ...] = ("step", "ramp", "noise", "pulses")
OOD_FAMILIES: tuple[str, ...] = ("strong_step", "chirp")
ALL_FAMILIES: tuple[str, ...] = ID_FAMILIES + OOD_FAMILIES


@dataclass(frozen=True)
class StimulusConfig:
    """Ranges (uniform sampling) of the stimulus protocol parameters."""

    onset_ms: Range = (5.0, 20.0)
    step_amp: Range = (-8.0, 20.0)
    step_duration_ms: Range = (20.0, 90.0)
    ramp_peak: Range = (0.0, 30.0)
    noise_mean: Range = (-2.0, 10.0)
    noise_std: Range = (0.0, 8.0)
    noise_tau_ms: Range = (1.0, 10.0)
    pulse_amp: Range = (5.0, 60.0)
    pulse_width_ms: Range = (0.5, 2.0)
    pulse_rate_hz: Range = (10.0, 150.0)
    strong_step_amp: Range = (25.0, 60.0)
    chirp_offset: Range = (0.0, 6.0)
    chirp_amp: Range = (4.0, 15.0)
    chirp_f_hz: Range = (2.0, 200.0)
    id_weights: dict[str, float] = field(
        default_factory=lambda: {"step": 0.3, "ramp": 0.15, "noise": 0.35, "pulses": 0.2}
    )


def _u(rng: np.random.Generator, r: Range) -> float:
    return float(rng.uniform(r[0], r[1]))


def make_stimulus(
    family: str,
    rng: np.random.Generator,
    n_steps: int,
    dt: float,
    cfg: StimulusConfig | None = None,
) -> FloatArray:
    """Sample one current waveform of ``family`` with ``n_steps`` samples."""
    cfg = cfg or StimulusConfig()
    t = dt * np.arange(n_steps)
    t_max = n_steps * dt
    onset = min(_u(rng, cfg.onset_ms), 0.5 * t_max)
    i = np.zeros(n_steps, dtype=np.float64)
    after = t >= onset

    if family in ("step", "strong_step"):
        amp = _u(rng, cfg.step_amp if family == "step" else cfg.strong_step_amp)
        offset = onset + _u(rng, cfg.step_duration_ms)
        i[after & (t < offset)] = amp
    elif family == "ramp":
        peak = _u(rng, cfg.ramp_peak)
        i[after] = peak * (t[after] - onset) / max(t_max - onset, dt)
    elif family == "noise":
        mean = _u(rng, cfg.noise_mean)
        std = _u(rng, cfg.noise_std)
        tau = _u(rng, cfg.noise_tau_ms)
        # Exact OU update on the sampling grid (stationary from the start).
        a = np.exp(-dt / tau)
        b = std * np.sqrt(1.0 - a * a)
        x = np.empty(n_steps)
        x[0] = std * rng.standard_normal()
        xi = rng.standard_normal(n_steps)
        for k in range(1, n_steps):
            x[k] = a * x[k - 1] + b * xi[k]
        i[after] = mean + x[after]
    elif family == "pulses":
        amp = _u(rng, cfg.pulse_amp)
        width = _u(rng, cfg.pulse_width_ms)
        period = 1000.0 / _u(rng, cfg.pulse_rate_hz)
        phase = np.mod(t - onset, period)
        i[after & (phase < width)] = amp
    elif family == "chirp":
        offset = _u(rng, cfg.chirp_offset)
        amp = _u(rng, cfg.chirp_amp)
        f0, f1 = cfg.chirp_f_hz
        tau = np.clip(t - onset, 0.0, None)
        duration = max(t_max - onset, dt)
        # Linear frequency sweep f(t) = f0 + (f1 - f0) t / T (t in s).
        phase = 2.0 * np.pi * (f0 * tau + 0.5 * (f1 - f0) * tau**2 / duration) / 1000.0
        i[after] = offset + amp * np.sin(phase[after])
    else:
        raise ValueError(f"unknown stimulus family {family!r}")
    return i


def sample_families(
    rng: np.random.Generator, n: int, ood: bool, cfg: StimulusConfig | None = None
) -> list[str]:
    """Draw ``n`` family labels (ID: weighted by ``cfg.id_weights``)."""
    cfg = cfg or StimulusConfig()
    if ood:
        return [str(f) for f in rng.choice(OOD_FAMILIES, size=n)]
    names = list(cfg.id_weights)
    w = np.array([cfg.id_weights[k] for k in names], dtype=np.float64)
    return [str(f) for f in rng.choice(names, size=n, p=w / w.sum())]
