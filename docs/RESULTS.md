# Results

Every number below comes from one run of the **full** configuration
([`configs/full.yaml`](../configs/full.yaml)):

* **Hardware:** a 4-core x86-64 CPU, no GPU; Python 3.13, PyTorch 2.15, numpy 2.5.
* **Runtime:** 47.5 min wall-clock, including data simulation, training of
  both candidate architectures and the 5-member ensemble, evaluation and
  figures.
* **Sources:** the raw numbers are in
  [`results/full/metrics.json`](../results/full/metrics.json), the
  auto-generated tables in [`results/full/report.md`](../results/full/report.md)
  and the training log in [`results/full/run.log`](../results/full/run.log).

For the default `small` configuration that CI runs, see §8. For the biology,
numerics and metric definitions, see [SCIENCE.md](SCIENCE.md).

```bash
python -m neurotwin run --config configs/full.yaml --out results/full --figdir docs/figures
```

## Summary

| question | answer |
|---|---|
| Which surrogate? | **An ensemble of 5 patch-wise GRUs.** A state-space neural ODE is equally (slightly more) accurate but no faster than integrating the real equations in batch (§2). |
| Accuracy on held-out parameters (in range) | Spike coincidence **Γ = 0.93** (median 1.0); **88 %** of traces have the exact spike count; median voltage RMSE **1.7 mV** (0.5 mV sub-threshold); median first-spike latency error **0.02 ms**. |
| Accuracy when extrapolating | Parameters outside the training box: Γ = 0.90. Novel stimuli: Γ = 0.76, ranging from 0.97 for chirps to 0.47 for strong steps. |
| Speed-up | One GRU is **167×** faster than scipy LSODA (0.54 vs 90 ms per 100 ms trace, batch 1 024, 4 threads). The 5-member ensemble with uncertainty is **26×** faster. Against a vectorised numpy RK4 integrator of the same equations, the gains are only **7×** and **1.1×**. |
| Does it know when it is extrapolating? | Mostly. Ensemble spread ranks trace errors with Spearman ρ = 0.90. Flagged traces have mean Γ 0.51 against 0.97 for unflagged ones. Spread detects novel stimuli (AUROC 0.82) much better than extrapolated parameters (0.64), so the twin pairs it with an exact parameter-range check. |
| Main failure modes | Sustained drive far beyond the training range (the f-I curve collapses at 45 µA/cm²); confident errors on near-threshold inputs; ramps; spike-time drift late in the trace (§6). |

## 1. Data

| split | traces | parameters | stimuli |
|---|---|---|---|
| train | 4 000 | training box | step, ramp, OU noise, pulse train |
| validation | 300 | training box (new draws) | same families |
| test_id | 300 | training box (new draws) | same families |
| test_ood_param | 300 | wider box, outside the training box in ≥ 1 parameter | same families |
| test_ood_stim | 300 | training box (new draws) | strong steps (25–60 µA/cm²), chirps 2→200 Hz |

* **Trace format:** 100 ms at 0.1 ms resolution, i.e. 1 000 samples per trace.
* **Ground truth:** HH equations integrated by RK4 at 0.02 ms, with a stiff
  LSODA fallback that was needed for 1 of the 5 200 traces. Agreement with
  LSODA at rtol = 1e-9 on 50 test traces:
  * max |ΔV| = 0.016 mV;
  * RMSE = 3.4·10⁻⁴ mV;
  * identical spike trains (Γ = 1.000).

  The simulator's numerical error (RMSE 3·10⁻⁴ mV) is therefore three orders
  of magnitude below the surrogate errors reported here.
* **Simulation cost:** 32 s on 4 cores for all 5 200 traces.

## 2. Choosing the architecture: GRU vs neural ODE

![model comparison](figures/model_comparison.png)

Both candidates were trained on the same data with about the same wall-clock
budget: 1 244 s for the GRU and 1 306 s for the neural ODE, each on 2 CPU
threads.

**Validation set (300 traces):**

| model | weights | mean Γ | exact spike count | median RMSE (mV) | mean RMSE (mV) | median sub-thr. RMSE (mV) | batched speed-up vs scipy |
|---|---|---|---|---|---|---|---|
| GRU (2×128, 0.5 ms patches) | 203 k | 0.948 | 0.880 | 1.65 | 2.66 | 0.47 | 120× |
| neural ODE (MLP 4×128, RK4 at 0.05 ms) | 36 k | 0.949 | **0.953** | 1.56 | 3.28 | **0.28** | 16× |

**Held-out in-range test set, single model each:**

| model | mean Γ | exact spike count | median RMSE (mV) |
|---|---|---|---|
| GRU | 0.928 | 0.887 | 1.80 |
| neural ODE | **0.943** | **0.937** | 1.78 |

**Accuracy.** With 4 000 training traces the neural ODE is at least as
accurate as the GRU, and slightly better on spike counts and sub-threshold
voltage:

* It reaches this with 6× fewer weights.
* It has privileged access to the gating variables (m, h, n) during training.
* It has a strong inductive bias: it only learns a smooth local vector field.
* Its *mean* RMSE is worse than the GRU's, so its rare errors are larger. When
  a rollout drifts, it drifts further.

The picture is different with little data. On the small configuration (800
traces) the GRU is clearly better: Γ 0.86 vs 0.70 (§8).

**Speed.** The neural ODE inherits the stiffness of the HH equations:

* RK4 on the true equations diverges at 0.1 ms, so the neural ODE must step
  at 0.05 ms.
* That costs 2 000 MLP evaluations per trace, against 200 GRU steps.
* Batched, it costs 4.15 ms per trace. Integrating the *real* HH equations
  with the same batched RK4 approach in numpy costs 3.88 ms.
* For a single trace it is 6× slower than scipy (563 vs 90 ms).

A neural ODE in the native state space therefore gives no computational
advantage over the model it emulates. The GRU learns the flow map over
0.5 ms steps and is 7.7× faster batched and 50× faster for single traces.

**Decision rule.** Among candidates whose validation Γ is within 0.01 of the
best, choose the fastest. That makes the GRU the twin. If raw accuracy per
weight matters more than speed (for example, offline fitting with a GPU-free
budget), the neural ODE is the better emulator. `models.NeuralODESurrogate`
is kept for that purpose.

> **How the rule evolved (for transparency).** The selection rule was revised
> twice during this study.
>
> 1. The first full run selected on validation Γ alone. With a 40-epoch GRU
>    (less training time than the neural ODE), the neural ODE won: Γ 0.954 vs
>    0.934.
> 2. Because the brief requires the twin to be far faster than the
>    integrator, the rule became "most accurate model with ≥ 100× speed-up".
>    The GRU also received an equal training budget of 60 epochs. The two
>    then tied on accuracy, and the GRU measured 98×, against 120× in the
>    final run. A hard gate on so noisy a measurement is fragile.
> 3. The rule therefore became the tie-aware version above. Under it, the
>    GRU is selected whether its measured speed-up is 98× or 120×.

## 3. Accuracy on held-out parameters

![true vs surrogate traces](figures/traces_true_vs_surrogate.png)

*Each row shows a typical trace (median error) and a bad one (90th-percentile
error) from one test set. Grey: injected current. Black: HH simulator. Blue
dashed: twin (ensemble mean). Shaded: ±2σ ensemble spread.*

| split | spiking traces | mean Γ | median Γ | exact spike count | spike-count R² | median RMSE (mV) | median sub-thr. RMSE (mV) | median 1st-spike latency error (ms) |
|---|---|---|---|---|---|---|---|---|
| test_id (in range) | 85 % | **0.934** | 1.000 | **0.877** | 0.911 | **1.70** | 0.48 | 0.02 |
| test_ood_param (extrapolated θ) | 80 % | 0.900 | 1.000 | 0.850 | 0.779 | 2.60 | 0.59 | 0.06 |
| test_ood_stim (novel stimuli) | 100 % | 0.755 | 1.000 | 0.593 | −1.95 | 3.71 | 0.71 | 0.02 |

**Summary features** (mean absolute error), computed from the predicted
traces:

| split | firing rate (Hz) | first-spike latency (ms) | mean sub-thr. V (mV) | peak V (mV) |
|---|---|---|---|---|
| test_id | 2.5 | 0.71 | 0.48 | 4.6 |
| test_ood_param | 4.0 | 1.94 | 1.05 | 6.4 |
| test_ood_stim | 17.2 | 0.09 | 1.42 | 2.1 |

**Spike timing by stimulus family** (mean Γ; * = never seen in training):

| noise | pulses | ramp | step | chirp* | strong step* |
|---|---|---|---|---|---|
| 0.969 | 0.942 | 0.761 | 0.983 | 0.973 | 0.474 |

**In range.**

* The twin reproduces spike times to within the 2 ms coincidence window on
  the large majority of traces. The median trace is perfect (Γ = 1).
* The first spike is placed with a median error of 20 µs.
* The sub-threshold membrane potential is accurate to about 0.5 mV.

**Extrapolated parameters.**

* Accuracy degrades gracefully: Γ drops from 0.93 to 0.90 and median RMSE
  rises from 1.7 to 2.6 mV.
* The decline concentrates where extrapolation makes the cell *more*
  excitable: high $\bar g_{Na}$ and low $\bar g_K$ (figure below). There, the
  binned median RMSE climbs to ~10 mV at the edges of the wider box.
* Changes of $g_L$ and $C_m$ beyond the training range are tolerated well.

**Novel stimuli.**

* The twin generalises to chirps (Γ 0.97), which recombine frequencies and
  amplitudes it has seen.
* It fails on sustained strong steps (Γ 0.47). It fires the first spike
  correctly and then under-fires; see §6.

![error vs parameter](figures/error_vs_parameter.png)

*Per-trace RMSE (top) and ensemble spread (bottom) against each parameter.
The grey band is the training range; black lines are binned medians. The
right-most column pools all parameters as the distance outside the training
box, in units of the box half-width.*

**Ensemble mean vs a single member.**

| split | mean Γ, ensemble mean | mean Γ, single member | median RMSE, ensemble mean (mV) | median RMSE, single member (mV) |
|---|---|---|---|---|
| test_id | 0.934 | 0.939 | 1.70 | 1.84 |
| test_ood_param | 0.900 | 0.899 | 2.60 | 2.69 |
| test_ood_stim | 0.755 | 0.804 | 3.71 | 3.91 |

Averaging the members lowers the voltage RMSE. It also slightly *lowers*
spike metrics where members disagree on timing, because the averaged trace
smears or shrinks the disputed spike. Downstream spike statistics should
therefore be computed per member and then averaged, which is how the f-I
curves below are made.

![f-I curves](figures/fi_curves.png)

**f-I curves** (90 ms steps):

* The twin reproduces the f-I curves of the classic neuron and of two corners
  of the training box up to 30 µA/cm². That is beyond the 20 µA/cm² maximum
  of the training steps; training ramps reach 30 µA/cm², but only at the end
  of a trace. That includes the discontinuous class-II onset at ≈6–8
  µA/cm² and the near-silent "single-spike" corner (low $\bar g_{Na}$, high
  $\bar g_K$).
* It errs at 6 µA/cm² for the classic cell (11 vs 22 Hz: one spike instead
  of two). This is just below the onset of repetitive firing (≈6.3 µA/cm²),
  where the response is most sensitive.
* It collapses at 45 µA/cm² (classic: 18 vs 111 Hz). The members disagree
  strongly there (error bars).
* Mean absolute rate error over all points: 7.0 Hz.

## 4. Speed-up

![speedup](figures/speedup.png)

Seconds of CPU wall-clock per simulated 100 ms trace (1 000 output samples):

| method | batch | threads | ms / trace | vs scipy LSODA | vs numpy RK4 |
|---|---|---|---|---|---|
| scipy LSODA, one trace at a time (reference) | 1 | 1 | 90.1 | 1× | 0.04× |
| numpy RK4, vectorised, 0.02 ms step | 1 024 | 1 | 3.88 | 23× | 1× |
| neural ODE | 1 024 | 4 | 4.15 | 22× | 0.93× |
| neural ODE, single trace | 1 | 4 | 618 | 0.15× | — |
| **GRU, single trace** | 1 | 1 | 11.2 | **8×** | — |
| GRU | 64 | 4 | 0.93 | 97× | 4.2× |
| GRU | 1 024 | 1 | 1.76 | 51× | 2.2× |
| **GRU** | 1 024 | 4 | **0.54** | **167×** | **7.2×** |
| **GRU ensemble ×5 (with uncertainty)** | 1 024 | 4 | 3.45 | **26×** | 1.1× |

How to read these numbers:

* **Against the usual workflow, a serial adaptive solver, one GRU is 167×
  faster.** That is within the 100–1 000× range typically quoted for neural
  surrogates. In practice: 10 000 traces take ~5 s instead of ~15 min.
* **Much of that gain comes from batching, not from the network.** A
  carefully vectorised fixed-step integrator of the *true* equations is
  itself 23× faster than LSODA. Against it, one GRU is 7× faster and the
  5-member ensemble breaks even. Speed-up claims for neural surrogates should
  always state the baseline.
* **Single-trace latency is where the twin helps least:** 11 ms vs 90 ms (8×).
  It is still the fastest option for one trace. Fixed-step numpy RK4 has
  large per-step overhead at batch size 1 and takes 1.5 s per trace.
* **Untested routes to more speed:**
  * a GPU, which would favour large-batch GRU inference more than the
    CPU-bound solvers;
  * larger patches (fewer recurrent steps);
  * distilling the ensemble into a single network that predicts mean and
    spread.

## 5. Uncertainty

![uncertainty](figures/uncertainty.png)

The extrapolation flag fires when the time-averaged ensemble spread
$\bar\sigma$ exceeds 1.23 mV, the 95th percentile on the validation set.

| split | median $\bar\sigma$ (mV) | ±2σ coverage | Spearman($\bar\sigma$, RMSE) | flagged by σ | flagged by σ or range check |
|---|---|---|---|---|---|
| test_id | 0.43 | 0.73 | 0.91 | 9.7 % | 9.7 % |
| test_ood_param | 0.61 | 0.71 | 0.89 | 24.3 % | 100 % |
| test_ood_stim | 0.87 | 0.73 | 0.86 | 35.7 % | 35.7 % |

* **The spread is a good error ranker.** Pooled over all test sets, Spearman
  ρ($\bar\sigma$, RMSE) = 0.90.
* **The flag separates good from bad predictions:**

  | | mean Γ | median RMSE |
  |---|---|---|
  | flagged traces | 0.51 | 11.9 mV |
  | unflagged traces | 0.97 | 2.0 mV |

  Across the three test sets the flag fires on 23 % of traces, yet it catches
  90 % of the traces with RMSE > 10 mV and 71 % of those with RMSE > 5 mV.
  Re-simulating the flagged traces with the simulator therefore removes most
  of the large errors at a quarter of the simulation cost.
* **It detects novel stimuli better than novel parameters.** AUROC for
  separating in-range from novel-stimulus traces is 0.82. For extrapolated
  parameters it is only 0.64, and the spread correlates only weakly with the
  distance outside the training box (ρ = 0.17). The reason is that most
  extrapolated-parameter traces are predicted *well* (Γ 0.90), so their low
  uncertainty is correct. The spread tracks *error*, not novelty per se.
  Parameter novelty is caught exactly and for free by a range check: every θ
  is normalised so the training box is [−1, 1]⁴. The twin therefore reports
  both.
* **Intervals are not calibrated.** Only ~72 % of samples fall within ±2σ,
  against a nominal 95 %. Five members underestimate the spread everywhere,
  and most of all around spikes: coverage is 55 % within −1 to +4 ms of a
  spike and 77 % elsewhere. Near a spike, a 0.2 ms timing error is a 50 mV
  voltage error. Use σ to rank and flag, not as a likelihood.
* **The flag rate on in-range test data is 9.7 %.** The 95th-percentile
  threshold was set on 300 validation traces, so the nominal 5 % is only
  approximate.

## 6. Failure modes

![failure modes](figures/failure_modes.png)

1. **Sustained drive beyond the training range.**
   * Steps of 25–60 µA/cm² give Γ = 0.47 and spike-count R² < 0.
   * The twin fires the first spike correctly, then produces damped,
     sub-threshold oscillations instead of a high-rate spike train (§3,
     traces; f-I at 45 µA/cm²).
   * The network has not learned the high-rate limit cycle it never saw.
   * The ensemble spread does rise sharply (flag rate 36 %), so the failure
     is visible.
2. **Near-threshold inputs, sometimes with confident errors.**
   * The worst in-range trace (right panel, RMSE 30 mV) is a 23.6 µA/cm²
     pulse train at the threshold amplitude.
   * The simulator answers some pulses with delayed spikes (latency up to
     4 ms) and skips others. The twin picks a different subset of pulses.
   * Its spread is 0.8 mV, below the 1.23 mV flag: all members make the same
     mistake, so the error is confident. A deep ensemble cannot protect
     against this, because its members share the training data and its
     blind spots.
   * Threshold inputs are intrinsically ill-conditioned: an infinitesimal
     input change flips spike/no-spike. Hodgkin-Huxley's bistability between
     ≈6.3 and ≈9.8 µA/cm² (SCIENCE §6) makes the step and ramp regime around
     it the hardest.
3. **Ramps.**
   * They are the weakest training family (mean Γ 0.76).
   * In the ramp's threshold region, the stimulus is a slowly rising current,
     so the moment of the first spike is very sensitive to small errors
     (accommodation).
   * Every later spike inherits the shift.
4. **Spike-time drift.**
   * The in-range error grows from ≈0.5 mV just after stimulus onset to
     ≈2 mV at 100 ms (middle panel).
   * Once one spike is late, everything after it is shifted.
   * Longer simulations would need periodic re-anchoring to the simulator,
     or training on longer windows.
5. **Excitable extrapolations.**
   * High $\bar g_{Na}$ (> 150) or low $\bar g_K$ (< 25) make the cell much
     more excitable, though not spontaneously active.
   * A 3 µA/cm² step evokes 5–6 spikes instead of the single spike of the
     classic cell.
   * Error and spread rise together (§3 figure).
6. **Blurred ensemble mean.**
   * Where members disagree, their average lowers and widens the disputed
     spikes (§3).
   * This is not a failure of the uncertainty, but it means "the ensemble
     mean trace" should not be fed to a spike detector without care.

## 7. Reproducibility

* All randomness is seeded. Re-running the full configuration reproduced the
  same candidate-training trajectories epoch by epoch (identical losses in
  `run.log`).
* The simulated datasets and the trained models are cached in
  `results/<config>/data` and `results/<config>/models`. Neither is committed;
  both regenerate from the config.
* The trained 5-member twin is committed in `results/full/twin` (4 MB).
  `tests/test_shipped_twin.py` checks it against the simulator on every CI
  run.
* The figures in `docs/figures/` are written by the same command.

## 8. Small (default, CI) configuration

The default configuration ([`configs/small.yaml`](../configs/small.yaml)) is
what CI runs end-to-end on every pull request.

| | small | full |
|---|---|---|
| training traces / test traces per split | 800 / 100 | 4 000 / 300 |
| GRU (hidden, epochs) / ensemble members | 64, 20 / 3 | 128, 60 / 5 |
| validation Γ: GRU / neural ODE | 0.856 / 0.702 | 0.948 / 0.949 |
| validation median RMSE: GRU / neural ODE (mV) | 3.42 / 11.05 | 1.65 / 1.56 |
| selected | GRU | GRU |
| test_id: Γ / exact count / median RMSE (mV) | 0.890 / 0.70 / 3.91 | 0.934 / 0.88 / 1.70 |
| test_ood_param: Γ / median RMSE (mV) | 0.765 / 5.12 | 0.900 / 2.60 |
| test_ood_stim: Γ / median RMSE (mV) | 0.568 / 10.17 | 0.755 / 3.71 |
| AUROC σ: ID vs extrapolated params / novel stimuli | 0.62 / 0.88 | 0.64 / 0.82 |
| scipy LSODA (ms / trace) | 124 | 90 |
| one GRU, largest batch, 4 threads: ms / trace (speed-up) | 0.52 (237×, batch 256) | 0.54 (167×, batch 1 024) |
| ensemble, largest batch, 4 threads (speed-up) | 90× | 26× |
| pipeline wall-clock (4-core CPU) | 6.4 min | 47.5 min |

The trends with more data and capacity:

* Accuracy improves across the board: in-range Γ rises from 0.89 to 0.93
  and median RMSE falls from 3.9 to 1.7 mV.
* The neural ODE improves most. It is clearly worse than the GRU with 800
  traces and catches up with 4 000.
* The two GRUs cost about the same per trace (0.52 vs 0.54 ms). The higher
  speed-ups of the small configuration come from a slower scipy measurement
  in that run and, for the ensemble, from 3 smaller members instead of 5.

Speed measurements vary by about ±20 % between runs on the same machine. For
example, scipy took 100 ms per trace in one small run and 124 ms in another.
Treat speed-up factors as accurate to that level.

The numbers for this configuration are in
[`results/small/metrics.json`](../results/small/metrics.json) and
[`results/small/report.md`](../results/small/report.md). CI uploads the same
files, plus figures, as the `small-results` artifact of every run.
