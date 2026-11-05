# Results tables (small config)

## Architecture selection (validation split, single model each)

| model | params | train time (s) | mean Gamma | exact spike count | median RMSE (mV) | median sub-threshold RMSE (mV) | speed-up vs scipy |
|---|---|---|---|---|---|---|---|
| GRU | 53,770 | 159 | 0.837 | 0.700 | 4.51 | 1.27 | 299x |
| neural ODE | 5,640 | 78 | 0.702 | 0.580 | 11.05 | 2.73 | 10x |

Rule: candidates meeting the speed-up requirement (all, if none does) whose validation mean Gamma is within 0.01 of the best are equally accurate; among them the fastest is chosen (requirement: batched speed-up >= 100x vs scipy LSODA). Selected: **GRU**.

## Held-out accuracy of the twin (ensemble mean)

| split | traces | spiking | mean Gamma | median Gamma | exact spike count | spike-count R^2 | median RMSE (mV) | median sub-thr. RMSE (mV) | 1st-spike latency err (ms) |
|---|---|---|---|---|---|---|---|---|---|
| test_id | 100 | 0.79 | 0.901 | 1.000 | 0.750 | 0.790 | 4.20 | 1.17 | 0.05 |
| test_ood_param | 100 | 0.76 | 0.763 | 1.000 | 0.650 | 0.329 | 5.48 | 1.71 | 0.11 |
| test_ood_stim | 100 | 1.00 | 0.554 | 0.458 | 0.310 | -3.093 | 10.76 | 3.70 | 0.05 |

Single ensemble member (average over members), same splits:

| split | mean Gamma | exact spike count | median RMSE (mV) |
|---|---|---|---|
| test_id | 0.898 | 0.747 | 4.34 |
| test_ood_param | 0.776 | 0.670 | 5.83 |
| test_ood_stim | 0.559 | 0.313 | 10.38 |

Summary-feature errors (mean absolute error, ensemble mean trace):

| split | firing rate (Hz) | first-spike latency (ms) | mean sub-thr. V (mV) | peak V (mV) |
|---|---|---|---|---|
| test_id | 5.10 | 2.54 | 1.02 | 6.69 |
| test_ood_param | 11.30 | 2.15 | 2.36 | 14.47 |
| test_ood_stim | 27.40 | 0.30 | 3.53 | 4.50 |

Mean Gamma by stimulus family (* = never seen in training):

| noise | pulses | ramp | step | chirp* | strong_step* |
|---|---|---|---|---|---|
| 0.908 | 0.908 | 0.848 | 0.914 | 0.836 | 0.283 |

## Uncertainty

| split | median sigma (mV) | 2-sigma coverage | Spearman(sigma, RMSE) | flagged as extrapolation |
|---|---|---|---|---|
| test_id | 0.69 | 0.532 | 0.851 | 0.050 |
| test_ood_param | 0.89 | 0.547 | 0.905 | 0.290 |
| test_ood_stim | 1.81 | 0.542 | 0.754 | 0.520 |

- flag rate when sigma is combined with a parameter range check: test_id 0.050, test_ood_param 1.000, test_ood_stim 0.520
- flag threshold (95th pct of validation sigma): 1.71 mV
- AUROC ID vs extrapolated params: 0.644; ID vs novel stimuli: 0.873
- Spearman(sigma, RMSE), pooled test splits: 0.885
- Spearman(sigma, distance outside training box): -0.061
- mean Gamma flagged / unflagged: 0.358 / 0.892; median RMSE flagged / unflagged: 15.05 / 4.46 mV

## Speed (seconds per 100 ms trace, CPU)

| method | batch | threads | ms / trace | speed-up vs scipy | speed-up vs numpy RK4 |
|---|---|---|---|---|---|
| scipy_lsoda | 1 | 1 | 134 | 1.0 | 0.09 |
| numpy_rk4_batch | 256 | 1 | 11.52 | 11.6 | 1.00 |
| GRU_b1_t1 | 1 | 1 | 8.242 | 16.3 | 1.40 |
| GRU_b256_t1 | 256 | 1 | 0.337 | 398.0 | 34.22 |
| neural ODE_b1_t1 | 1 | 1 | 456.1 | 0.3 | 0.03 |
| neural ODE_b256_t1 | 256 | 1 | 5.807 | 23.1 | 1.98 |
| ensemble3_b256_t1 | 256 | 1 | 0.887 | 151.1 | 13.00 |
| GRU_b1_t4 | 1 | 4 | 6.545 | 20.5 | 1.76 |
| GRU_b256_t4 | 256 | 4 | 0.216 | 620.4 | 53.34 |
| neural ODE_b1_t4 | 1 | 4 | 517.4 | 0.3 | 0.02 |
| neural ODE_b256_t4 | 256 | 4 | 6.615 | 20.3 | 1.74 |
| ensemble3_b256_t4 | 256 | 4 | 0.617 | 217.2 | 18.67 |

Reference check: dataset integrator (RK4, dt/5) vs scipy LSODA (rtol 1e-9) on 20 test traces: max |dV| = 7.92e-03 mV, RMSE = 1.56e-04 mV, mean Gamma = 1.000.

f-I curves: mean absolute rate error 21.6 Hz.

Total pipeline time: 323 s.
