# neuro-digital-twin

A fast, uncertainty-aware deep-learning **digital twin of a Hodgkin-Huxley
neuron**. A PyTorch surrogate learns the map (biophysical parameters,
injected current) → membrane-potential trace. It replaces the stiff ODE
integrator at a fraction of the cost and reports, through a deep ensemble, when
it is extrapolating.

* **Results** (accuracy on held-out parameters, speed-up, uncertainty, failure
  modes): [`docs/RESULTS.md`](docs/RESULTS.md)
* **Background for neuroscientists**, with references:
  [`docs/SCIENCE.md`](docs/SCIENCE.md)

![true vs surrogate traces](docs/figures/traces_true_vs_surrogate.png)

## Headline results

These come from the full config on a 4-core CPU; details are in
[`docs/RESULTS.md`](docs/RESULTS.md).

| | |
|---|---|
| Twin | ensemble of 5 patch-wise GRUs, conditioned on $\bar g_{Na}, \bar g_K, g_L, C_m$ |
| Accuracy, held-out parameters in the training range | spike coincidence Γ = 0.93; exact spike count in 88 % of traces; median RMSE 1.7 mV |
| Accuracy, extrapolated parameters / novel stimuli | Γ = 0.90 / 0.76 |
| Speed-up vs scipy LSODA | **167×** for one GRU and 26× for the 5-member ensemble (batch 1 024); 8× for a single trace |
| Speed-up vs a vectorised numpy RK4 of the same equations | 7× for one GRU; about 1× for the ensemble |
| Uncertainty | ensemble spread ranks errors (Spearman 0.90) and flags 90 % of traces with RMSE > 10 mV while flagging 23 % of traces |
| Faster twin with 1 ms GRU steps ([RESULTS §9](docs/RESULTS.md#9-larger-gru-time-steps-and-gpu-readiness)) | same in-range accuracy (Γ 0.94, RMSE 1.5 mV), **2× faster**: 319× for one GRU, 56× for the ensemble; slightly weaker extrapolation; shipped in `results/full_p10/twin` |
| Why a GRU and not a neural ODE? | The neural ODE is as accurate (better on some metrics), but it inherits HH stiffness and is no faster than integrating the real equations |

## What is in the box

| component | file | notes |
|---|---|---|
| Reference simulator | `src/neurotwin/hh.py` | Hodgkin-Huxley (1952) neuron with Q10 temperature scaling. Uses scipy LSODA as the reference and vectorised numpy RK4 (with a stiff fallback) to build datasets. |
| Stimuli | `src/neurotwin/stimuli.py` | Steps, ramps, Ornstein-Uhlenbeck noise and pulse trains (training); strong steps and chirps (never seen in training). |
| Datasets | `src/neurotwin/data.py` | (θ, I(t)) → (V, m, h, n)(t) pairs plus summary features. Splits: train, validation, in-range test, extrapolated parameters, novel stimuli. |
| Surrogates | `src/neurotwin/models.py` | Patch-wise **GRU** sequence model and a **neural ODE** in the HH state space. Both are trained and compared; the better one becomes a deep ensemble. |
| Training | `src/neurotwin/train.py` | GRU: backpropagation through the full trace. Neural ODE: multiple shooting with a window curriculum. |
| Evaluation | `src/neurotwin/evaluate.py`, `features.py` | Coincidence factor Γ, spike counts, latency, sub-threshold RMSE, f-I curves, uncertainty AUROC and calibration, and timing against scipy and numpy. |
| Pipeline / CLI | `src/neurotwin/pipeline.py`, `__main__.py` | One command runs the whole experiment and writes `metrics.json`, `report.md`, figures and checkpoints. |

## Quick start

No GPU, no API keys and no external data: everything is simulated locally.

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -e ".[dev]"
```

Run the small default experiment (about 8 minutes on a 4-core CPU):

```bash
python -m neurotwin run --config configs/small.yaml
```

Run the full experiment behind `docs/RESULTS.md` (about 1 hour on a 4-core CPU):

```bash
python -m neurotwin run --config configs/full.yaml --out results/full --figdir docs/figures
```

Each run writes the following to `results/<name>/`:

* `metrics.json`: every number in the report, plus the full config and
  environment.
* `report.md`: the results tables.
* `figures/`: the plots (unless `--figdir` points elsewhere).
* `twin/`: the trained ensemble.
* `data/`: the cached simulations.

### Running on a GPU

Training, inference and the benchmarks run on a CUDA GPU when PyTorch sees
one: the configs default to `device: auto`. You can also force the device. In
the commands below, `<config>.yaml` is any config, e.g. `configs/full.yaml`.

```bash
pip install torch   # a CUDA build of PyTorch, see pytorch.org
```

```bash
python -m neurotwin run --config configs/full.yaml --device cuda --out results/full_gpu
```

```bash
python -m neurotwin sweep --config configs/full.yaml --device cuda --out results/full_gpu --patches 5 10 20 50
```

On a GPU:

* The speed benchmark also times every surrogate on the GPU, at batch sizes 1,
  1 024 and 8 192. GPU timings include the copies between host and device.
* The sweep adds GPU columns.
* CPU timings are still measured, for comparison.
* Models trained on a GPU are cached separately from CPU-trained ones.

### Patch-size sweep

`neurotwin sweep` trains one GRU per patch length (samples per recurrent step).
Every model uses the same data, seed and number of epochs. The sweep reports
accuracy on every held-out split and inference speed. Run in the output
directory of a finished experiment, it reuses that run's datasets and models.

On the full config, 1 ms and 2 ms steps are as accurate as the default 0.5 ms
step and 2× and 5× faster; 5 ms steps lose accuracy. See
[RESULTS §9](docs/RESULTS.md#9-larger-gru-time-steps-and-gpu-readiness).
Config entries can be overridden on the command line, for example
`--set gru.patch=10`.

### Using a trained twin from Python

The 5-member ensemble from the full configuration is committed in
`results/full/twin` (4 MB). The faster 1 ms-step twin of RESULTS §9 is in
`results/full_p10/twin`, and both load the same way. They are valid for
parameters inside the training box and for 100 ms traces at 0.1 ms resolution
that start from rest.

```python
import numpy as np
from neurotwin.pipeline import load_twin

twin = load_twin("results/full/twin")          # trained twin shipped in this repo
theta = np.array([[120.0, 36.0, 0.3, 1.0]])    # g_Na, g_K, g_L (mS/cm^2), C_m (uF/cm^2)
current = np.zeros((1, 1000))                  # 100 ms at dt = 0.1 ms, uA/cm^2
current[0, 100:900] = 10.0                     # 80 ms step of 10 uA/cm^2
mean, sigma, members = twin.predict(theta, current)   # mV, shapes (1, 1000)
print("ensemble spread (mV):", sigma.mean())  # large spread -> re-check with the simulator
```

## Development

The tests run fully offline in about a minute on a 4-core CPU.

```bash
ruff check src tests
```

```bash
mypy
```

```bash
pytest
```

GitHub Actions (`.github/workflows/ci.yml`) has two jobs:

* lint, strict type-checking and tests on Python 3.11 and 3.12;
* an end-to-end run of the small configuration, whose results are uploaded as
  a build artifact.

## License

MIT © 2026 Cesar Daniel Reinoso. See [LICENSE](LICENSE).
