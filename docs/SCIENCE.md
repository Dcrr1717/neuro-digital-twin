# The science behind `neurotwin`

*A guide for neuroscientists: what is simulated, what the network learns, how
accuracy is judged, and where the twin can and cannot be trusted.*

---

## 1. Why build a "digital twin" of a neuron?

Conductance-based models are the main way to link ion-channel biophysics to
spiking. They are also expensive. Each simulation integrates a stiff system of
ordinary differential equations (ODEs) with an adaptive solver. Many workflows
need that simulation thousands to millions of times:

* **Parameter inference.** Fitting maximal conductances to recordings, by
  optimisation or by simulation-based inference (Cranmer et al. 2020;
  Gonçalves et al. 2020).
* **Degeneracy maps.** Exploring which parameter combinations give the same
  behaviour (Prinz et al. 2004; Marder & Goaillard 2006).
* **Closed-loop and real-time use.** For example, choosing stimuli on the fly.

A **surrogate** (or *emulator*) is a model trained to reproduce the simulator's
input-output map. Once trained, it answers in a fraction of the time. We call
it a *digital twin* when it also reports how much its answer can be trusted.
Beniaguev et al. (2021) showed that deep networks can emulate a detailed
cortical neuron. This project does the same for the canonical
Hodgkin-Huxley (HH) neuron. It treats the HH integrator as ground truth and
asks three questions:

1. How accurate is the emulation on parameters it has never seen?
2. How much faster is it?
3. Does it know when it is extrapolating?

## 2. The reference model

### 2.1 Equations

We use the single-compartment model of Hodgkin & Huxley (1952) in the modern
sign convention (rest ≈ −65 mV):

$$
C_m \frac{dV}{dt} = I(t) - \bar g_{Na} m^3 h (V - E_{Na}) - \bar g_K n^4 (V - E_K) - g_L (V - E_L)
$$

$$
\frac{dx}{dt} = \phi(T)\,[\alpha_x(V)(1-x) - \beta_x(V)\,x], \qquad x \in \{m, h, n\}
$$

The rate functions are the standard HH ones
([`src/neurotwin/hh.py`](../src/neurotwin/hh.py)). The removable singularities
of $\alpha_m$ at −40 mV and $\alpha_n$ at −55 mV are evaluated analytically.
Temperature enters through $\phi = 3^{(T-6.3)/10}$, the Q10 = 3 scaling used by
Hodgkin & Huxley. All results use the reference temperature of 6.3 °C. The
reversal potentials are $E_{Na}=50$, $E_K=-77$ and $E_L=-54.387$ mV.

| symbol | meaning | classic value | training range | extrapolation range |
|---|---|---|---|---|
| $\bar g_{Na}$ | maximal Na⁺ conductance | 120 mS/cm² | 100–140 | 70–170 |
| $\bar g_K$ | maximal K⁺ conductance | 36 mS/cm² | 28–44 | 18–54 |
| $g_L$ | leak conductance | 0.3 mS/cm² | 0.2–0.4 | 0.1–0.6 |
| $C_m$ | membrane capacitance | 1.0 µF/cm² | 0.85–1.15 | 0.7–1.3 |

The twin is conditioned on these four parameters. The three conductances span
the classic space of conductance-based degeneracy studies. $C_m$ rescales every
voltage time constant. Each trace starts from the resting equilibrium of its
own parameter set, found by solving $I_{ion}(V)=0$ with the gates at steady
state.

**Sanity checks against known HH behaviour (classic parameters, our simulator,
300 ms steps from rest):**

* One spike only for steps of ≈2.3–6.2 µA/cm².
* Repetitive firing from ≈6.3 µA/cm². It starts at a *non-zero* rate of
  ≈53 Hz, the hallmark of Hodgkin's (1948) class-II excitability, which arises
  near a subcritical Hopf bifurcation (Rinzel & Miller 1980; Rinzel &
  Ermentrout 1998).
* About 67 Hz at 10 µA/cm² and 120 Hz at 60 µA/cm².
* Anode-break (rebound) spikes after release from hyperpolarisation.

`tests/test_hh.py` checks these behaviours.

### 2.2 Numerics: what counts as "ground truth"

The HH system is **stiff**: its time constants span orders of magnitude. At
strongly hyperpolarised potentials the Na⁺ activation gate relaxes in
microseconds (τ_m ≈ 7 µs at −130 mV), while the slowest processes take
milliseconds.

* The **reference solver** is LSODA (Petzold 1983), via `scipy.integrate.odeint`.
  LSODA switches automatically between non-stiff (Adams) and stiff (BDF)
  methods. Critical time points sit at every stimulus sample, so the solver
  never steps across a discontinuity in the injected current. This is the
  conventional way to simulate the model and the baseline for all speed-up
  figures.
* The **datasets** are generated with classical 4th-order Runge-Kutta (RK4) at
  a 0.02 ms step, vectorised over hundreds of neurons in numpy. On held-out
  test traces it agrees with LSODA run at very tight tolerances
  (rtol = 1e-9) to about 0.01 mV, with identical spike trains (Γ = 1; see
  RESULTS). That is orders of magnitude below any surrogate error.
* Explicit RK4 is only conditionally stable. In the rare traces where it
  diverges, the trace is recomputed with stiff LSODA. This happens under
  deep hyperpolarisation by strongly negative noise. It affected 1 of 5 200
  traces in the full configuration (a training trace) and 2 of 1 200 in the
  small one (both with extrapolated, low-leak parameters).

The stimulus is a **zero-order hold** on a 0.1 ms grid: the current is constant
on each sampling interval, like the output of a digital stimulator. The
voltage is reported at the end of each interval, so the map from stimulus to
voltage is strictly causal.

### 2.3 Stimulus protocols

Every trace lasts 100 ms. The stimulus begins after 5–20 ms of baseline.

| family | what it probes | used for |
|---|---|---|
| step (−8 to 20 µA/cm²) | rheobase, repetitive firing, anode-break excitation | training |
| ramp (to 0–30 µA/cm²) | accommodation, threshold for tonic firing | training |
| Ornstein-Uhlenbeck noise (mean −2 to 10, SD 0–8, τ 1–10 ms) | in-vivo-like fluctuating drive (Mainen & Sejnowski 1995; Destexhe et al. 2001) | training |
| pulse trains (5–60 µA/cm², 0.5–2 ms, 10–150 Hz) | refractoriness, entrainment | training |
| strong steps (25–60 µA/cm²) | high-rate firing beyond the trained amplitude range | **test only** |
| chirp (2→200 Hz sinusoid) | frequency preference / resonance | **test only** |

### 2.4 Test design

The twin is evaluated on four held-out sets. None of their parameter vectors
appear in training.

* **validation**: same distribution as training. Used only for architecture
  selection, early stopping and calibrating the extrapolation flag.
* **test_id**: new parameter vectors inside the training box, trained stimulus
  families. This is *interpolation*.
* **test_ood_param**: parameter vectors drawn from the wider box and *outside*
  the training box in at least one parameter. This is *extrapolation in
  parameter space*.
* **test_ood_stim**: in-range parameters driven by the two unseen stimulus
  families. This is *extrapolation in input space*.

## 3. The surrogates

Both surrogates map (parameters θ, stimulus I(t)) to V(t). Before entering the
network, parameters are rescaled so that the training box becomes [−1, 1]⁴.
Any coordinate beyond ±1 therefore signals extrapolation directly.

### 3.1 Neural ODE in the HH state space

A neural ODE (Chen et al. 2018) learns the *vector field*:

$$\frac{d\mathbf x}{dt} = f_\phi(\mathbf x, I, \theta), \qquad \mathbf x = (V, m, h, n)$$

Here $f_\phi$ is a multilayer perceptron, integrated with fixed-step RK4. We
give it every advantage available:

* It sees the *full* state, including the gating variables, which an
  experimentalist cannot measure.
* It is trained by **multiple shooting**: short rollouts of 8 → 32 → 96
  samples start from true states at random times, with a curriculum from short
  to long windows. This is the standard remedy for exploding gradients in long
  rollouts.
* Its initial (resting) state is learned from θ.

This is the "universal differential equation" approach of Rackauckas et al.
(2020). It is the natural choice if one wants a mechanistic-looking surrogate.

### 3.2 GRU sequence model

A gated recurrent unit (GRU; Cho et al. 2014) learns the **flow map over a
finite step** instead of the instantaneous vector field:

* The GRU reads the stimulus in *patches* of 5 samples (0.5 ms), with θ
  appended to every input.
* A small network maps θ to the initial hidden state, which plays the role of
  the parameter-dependent resting state.
* A read-out emits the 5 voltage samples of each patch.
* Training uses backpropagation through the whole 100 ms trace (200 recurrent
  steps), with an MSE loss on voltage only.

**Patch length.** The patch length is a free design choice. A sweep
(RESULTS §9) shows that 1–2 ms patches are as accurate as 0.5 ms patches and
2–5× faster; at 5 ms, accuracy degrades. Two candidate explanations, neither
tested here:

* fewer recurrent steps shorten back-propagation through time;
* a 1–2 ms patch spans about one action potential, so a single hidden state
  can emit the whole waveform.

### 3.3 Choosing between them

Both candidates are trained on the same data with about the same training
wall-clock (~21 min each on 2 CPU threads in the full configuration). The
twin exists to be fast, so the selection rule is:

1. Among candidates whose validation mean Γ is within 0.01 of the best, the
   accuracy is treated as equal.
2. Among those, the twin takes the one with the fastest batched inference.

The results ([RESULTS](RESULTS.md)) support three findings.

1. **With little data the GRU is clearly better.** With 800 training traces
   (small config), the GRU reaches Γ 0.86 against 0.70 for the neural ODE.
   Early in training, long neural-ODE rollouts even diverge.
2. **With 4 000 traces the neural ODE is at least as accurate.** On the
   primary metric they tie: validation Γ is 0.948 for the GRU and 0.949 for
   the neural ODE. On secondary metrics the neural ODE is better:
   * exact spike count: 0.95 vs 0.88;
   * sub-threshold RMSE: 0.28 vs 0.47 mV;
   * held-out test Γ: 0.943 vs 0.928.

   It gets there with 6× fewer weights. It benefits from privileged
   information (the gating variables) and from a strong inductive bias: it
   only has to learn a smooth, local vector field.
3. **The neural ODE is an order of magnitude slower, for a structural reason.**
   * A learned vector field reproduces the HH time scales, so an explicit
     integrator needs the same small steps as the original model. RK4 on the
     true HH equations diverges at 0.1 ms, so the neural ODE uses 0.05 ms.
   * That is 2 000 network evaluations per 100 ms trace, against 200 recurrent
     steps for the GRU.
   * Batched, the neural ODE is ~20× faster than scipy LSODA, and the GRU
     ~170×. That puts the neural ODE no faster than integrating the *true*
     HH equations with the same vectorised RK4 (4.2 vs 3.9 ms per trace),
     so it offers no computational benefit over the model it emulates. For
     a single trace it is 6× *slower* than scipy.
   * A discrete-time model escapes the constraint because it never integrates
     the fast dynamics. It only has to reproduce their net effect over each
     0.5 ms step.
   * Stiff neural ODEs are a known open problem (Kim et al. 2021; Kidger 2022).

The GRU is therefore the twin. The neural ODE still has merits:

* it is time-continuous and can be queried at any resolution;
* it exposes an interpretable state;
* it is the more data-efficient choice per weight once data are plentiful.

Implicit or exponential integrators could relax its speed limit. One example
is gates relaxing towards learned steady states, integrated with Rush-Larsen
(1978) steps.

**How this rule came about.** The rule was revised during the study, and both
earlier versions are recorded in RESULTS:

* First version, accuracy alone: the neural ODE won by a small margin over a
  GRU that had been given less training time.
* Second version, a hard 100× speed gate: the GRU then measured 98×, with
  timing noise of ±20 %, and was rejected.

## 4. Measuring accuracy the way electrophysiologists do

Voltage RMSE alone is misleading for spiking traces. A spike shifted by 1 ms
is penalised twice: once for the missing spike and once for the extra one.
That double penalty can be larger than the penalty for omitting the spike
altogether. We therefore report several measures:

* **Coincidence factor Γ** (Kistler et al. 1997; Jolivet et al. 2008):
  $$\Gamma = \frac{N_{coinc} - 2\nu\Delta N_{data}}{\tfrac12(N_{data}+N_{model})}\cdot\frac{1}{1-2\nu\Delta}$$
  * $N_{coinc}$ counts one-to-one matched spikes within $\Delta = 2$ ms.
  * $\nu$ is the surrogate's firing rate.
  * Γ = 1 means identical spike trains; Γ ≈ 0 means no better than a Poisson
    train with the same rate.
  * Two silent traces count as Γ = 1.
* **Spike count**: the fraction of traces with exactly the right count, and
  R² across traces.
* **First-spike latency error**.
* **Sub-threshold RMSE**: voltage error after excluding −1 to +4 ms around
  every true and predicted spike.
* **Summary features**: firing rate, latency, mean sub-threshold V and peak V
  computed from the predicted traces. This is what downstream analyses such
  as simulation-based inference typically consume.
* **f-I curves**: firing rate during a 90 ms step for the classic neuron and
  two corners of the parameter box.

Spikes are upward crossings of 0 mV. The detector re-arms only after the
voltage falls below −30 mV, so a jittery surrogate cannot score several
"spikes" on one action potential.

## 5. Uncertainty: does the twin know what it does not know?

We use a **deep ensemble** (Lakshminarayanan et al. 2017): 5 GRUs in the full
configuration (3 in the small one). The members differ only in random
initialisation and mini-batch order.

* The ensemble mean is the twin's prediction.
* The across-member standard deviation σ(t) is its **epistemic** uncertainty.
  The simulator is deterministic, so there is no aleatoric noise to model.
* In the training region the members agree because the data pin them down.
  Away from it, nothing constrains them to agree.

Deep ensembles are generally better calibrated under dataset shift than
MC-dropout (Gal & Ghahramani 2016; Ovadia et al. 2019), and they parallelise
trivially.

How we turn σ into a decision:

* The per-trace score is the time-averaged spread $\bar\sigma$.
* The **extrapolation flag** fires when $\bar\sigma$ exceeds the 95th
  percentile of its value on the validation set, so about 5 % of
  in-distribution traces are flagged by construction.
* We report how often the flag fires on each test set.
* We report the AUROC with which $\bar\sigma$ separates in-distribution from
  extrapolated cases.
* We report the rank correlation between $\bar\sigma$ and the actual error.

Two caveats:

* **Parameter extrapolation** is better detected by checking whether θ lies
  inside the training box. That check is exact and costs nothing. The value
  of σ lies elsewhere: it covers *novel stimuli*, which have no such simple
  test, and it flags *hard in-range cases*, such as inputs hovering near the
  firing threshold. The pipeline therefore also reports a combined flag:
  parameter range check OR high σ.
* **Pointwise intervals are not calibrated.** The ±2σ band is not a calibrated
  95 % interval. Spike-time errors are not Gaussian: a spike that occurs 1 ms
  late puts the truth far outside a narrow band for a short time. Use σ as a
  ranking or flagging score, not as a likelihood.

## 6. Known limits of the twin

* **Scope.** One compartment, four parameters, one temperature, 100 ms windows
  that start from rest. The twin does not know about dendrites, other channel
  types, intrinsic noise or longer histories. Retraining on a new simulator
  needs only the data-generation code to be swapped.
* **Threshold sensitivity is intrinsic.** Near rheobase, and in the bistable
  window between the onset of repetitive firing and the Hopf point (≈6.3–9.8
  µA/cm² for classic HH; Rinzel & Miller 1980), tiny input differences change
  the spike count. No surrogate can be uniformly accurate there. These are
  the traces where the twin's uncertainty is, appropriately, highest.
* **Spike-timing drift.** Errors grow over the course of a trace: once a spike
  is mistimed, everything after it is shifted.
* **Blurred ensemble mean.** Averaging members that disagree on spike times
  lowers peak amplitudes. Spike-based statistics are therefore also reported
  for individual members (RESULTS), and f-I rates are averaged per member.
* **Recommended use.** Use the twin for fast exploration, screening and
  inference. Re-simulate flagged cases, and any final result, with the
  reference simulator.

## References

* Beniaguev D, Segev I, London M (2021). Single cortical neurons as deep artificial neural networks. *Neuron* 109(17):2727–2739.
* Chen RTQ, Rubanova Y, Bettencourt J, Duvenaud D (2018). Neural ordinary differential equations. *Advances in Neural Information Processing Systems* 31.
* Cho K, van Merriënboer B, Gulcehre C, Bahdanau D, Bougares F, Schwenk H, Bengio Y (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *Proc. EMNLP*, 1724–1734.
* Cranmer K, Brehmer J, Louppe G (2020). The frontier of simulation-based inference. *PNAS* 117(48):30055–30062.
* Destexhe A, Rudolph M, Fellous J-M, Sejnowski TJ (2001). Fluctuating synaptic conductances recreate in vivo-like activity in neocortical neurons. *Neuroscience* 107(1):13–24.
* Gal Y, Ghahramani Z (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *Proc. ICML*, PMLR 48:1050–1059.
* Gerstner W, Kistler WM, Naud R, Paninski L (2014). *Neuronal Dynamics*. Cambridge University Press.
* Gonçalves PJ, Lueckmann J-M, Deistler M, et al. (2020). Training deep neural density estimators to identify mechanistic models of neural dynamics. *eLife* 9:e56261.
* Hairer E, Wanner G (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. Springer.
* Hodgkin AL (1948). The local electric changes associated with repetitive action in a non-medullated axon. *J Physiol* 107(2):165–181.
* Hodgkin AL, Huxley AF (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *J Physiol* 117(4):500–544.
* Izhikevich EM (2007). *Dynamical Systems in Neuroscience*. MIT Press.
* Jolivet R, Kobayashi R, Rauch A, Naud R, Shinomoto S, Gerstner W (2008). A benchmark test for a quantitative assessment of simple neuron models. *J Neurosci Methods* 169(2):417–424.
* Kidger P (2022). *On Neural Differential Equations*. PhD thesis, University of Oxford. arXiv:2202.02435.
* Kim S, Ji W, Deng S, Ma Y, Rackauckas C (2021). Stiff neural ordinary differential equations. *Chaos* 31:093122.
* Kistler WM, Gerstner W, van Hemmen JL (1997). Reduction of the Hodgkin-Huxley equations to a single-variable threshold model. *Neural Comput* 9(5):1015–1045.
* Lakshminarayanan B, Pritzel A, Blundell C (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems* 30.
* Mainen ZF, Sejnowski TJ (1995). Reliability of spike timing in neocortical neurons. *Science* 268(5216):1503–1506.
* Marder E, Goaillard J-M (2006). Variability, compensation and homeostasis in neuron and network function. *Nat Rev Neurosci* 7:563–574.
* Ovadia Y, Fertig E, Ren J, et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *Advances in Neural Information Processing Systems* 32.
* Petzold L (1983). Automatic selection of methods for solving stiff and nonstiff systems of ordinary differential equations. *SIAM J Sci Stat Comput* 4(1):136–148.
* Prinz AA, Bucher D, Marder E (2004). Similar network activity from disparate circuit parameters. *Nat Neurosci* 7(12):1345–1352.
* Rackauckas C, Ma Y, Martensen J, et al. (2020). Universal differential equations for scientific machine learning. arXiv:2001.04385.
* Rinzel J, Ermentrout GB (1998). Analysis of neural excitability and oscillations. In Koch C, Segev I (eds), *Methods in Neuronal Modeling*, 2nd ed. MIT Press.
* Rinzel J, Miller RN (1980). Numerical calculation of stable and unstable periodic solutions to the Hodgkin-Huxley equations. *Math Biosci* 49:27–59.
* Rush S, Larsen H (1978). A practical algorithm for solving dynamic membrane equations. *IEEE Trans Biomed Eng* 25(4):389–392.
