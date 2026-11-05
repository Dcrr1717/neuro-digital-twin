# Results tables (full config)

## Architecture selection (validation split, single model each)

| model | params | train time (s) | mean Gamma | exact spike count | median RMSE (mV) | median sub-threshold RMSE (mV) | speed-up vs scipy |
|---|---|---|---|---|---|---|---|
| GRU | 205,834 | 629 | 0.967 | 0.937 | 1.25 | 0.34 | 263x |
| neural ODE | 35,976 | 1306 | 0.949 | 0.953 | 1.56 | 0.28 | 18x |

Rule: candidates meeting the speed-up requirement (all, if none does) whose validation mean Gamma is within 0.01 of the best are equally accurate; among them the fastest is chosen (requirement: batched speed-up >= 100x vs scipy LSODA). Selected: **GRU**.

## Held-out accuracy of the twin (ensemble mean)

| split | traces | spiking | mean Gamma | median Gamma | exact spike count | spike-count R^2 | median RMSE (mV) | median sub-thr. RMSE (mV) | 1st-spike latency err (ms) |
|---|---|---|---|---|---|---|---|---|---|
| test_id | 300 | 0.85 | 0.938 | 1.000 | 0.873 | 0.902 | 1.47 | 0.34 | 0.01 |
| test_ood_param | 300 | 0.80 | 0.883 | 1.000 | 0.823 | 0.605 | 2.40 | 0.55 | 0.04 |
| test_ood_stim | 300 | 1.00 | 0.737 | 1.000 | 0.560 | -2.109 | 3.68 | 0.93 | 0.01 |

Single ensemble member (average over members), same splits:

| split | mean Gamma | exact spike count | median RMSE (mV) |
|---|---|---|---|
| test_id | 0.947 | 0.896 | 1.69 |
| test_ood_param | 0.895 | 0.883 | 2.69 |
| test_ood_stim | 0.779 | 0.635 | 4.06 |

Summary-feature errors (mean absolute error, ensemble mean trace):

| split | firing rate (Hz) | first-spike latency (ms) | mean sub-thr. V (mV) | peak V (mV) |
|---|---|---|---|---|
| test_id | 2.70 | 0.57 | 0.46 | 4.68 |
| test_ood_param | 6.27 | 1.66 | 1.42 | 7.62 |
| test_ood_stim | 18.40 | 0.09 | 1.77 | 2.24 |

Mean Gamma by stimulus family (* = never seen in training):

| noise | pulses | ramp | step | chirp* | strong_step* |
|---|---|---|---|---|---|
| 0.960 | 0.944 | 0.797 | 0.985 | 0.974 | 0.431 |

## Uncertainty

| split | median sigma (mV) | 2-sigma coverage | Spearman(sigma, RMSE) | flagged as extrapolation |
|---|---|---|---|---|
| test_id | 0.54 | 0.857 | 0.911 | 0.097 |
| test_ood_param | 0.70 | 0.789 | 0.925 | 0.240 |
| test_ood_stim | 1.17 | 0.835 | 0.940 | 0.437 |

- flag rate when sigma is combined with a parameter range check: test_id 0.097, test_ood_param 1.000, test_ood_stim 0.437
- flag threshold (95th pct of validation sigma): 1.62 mV
- AUROC ID vs extrapolated params: 0.624; ID vs novel stimuli: 0.808
- Spearman(sigma, RMSE), pooled test splits: 0.926
- Spearman(sigma, distance outside training box): 0.115
- mean Gamma flagged / unflagged: 0.497 / 0.976; median RMSE flagged / unflagged: 12.29 / 1.64 mV

## Speed (seconds per 100 ms trace, CPU)

| method | batch | threads | ms / trace | speed-up vs scipy | speed-up vs numpy RK4 |
|---|---|---|---|---|---|
| scipy_lsoda | 1 | 1 | 102.9 | 1.0 | 0.05 |
| numpy_rk4_batch | 1024 | 1 | 4.772 | 21.6 | 1.00 |
| GRU_b1_t1 | 1 | 1 | 9.679 | 10.6 | 0.49 |
| GRU_b64_t1 | 64 | 1 | 0.621 | 165.8 | 7.69 |
| GRU_b1024_t1 | 1024 | 1 | 0.878 | 117.2 | 5.44 |
| neural ODE_b1_t1 | 1 | 1 | 589.4 | 0.2 | 0.01 |
| neural ODE_b64_t1 | 64 | 1 | 22.39 | 4.6 | 0.21 |
| neural ODE_b1024_t1 | 1024 | 1 | 10.99 | 9.4 | 0.43 |
| ensemble5_b1024_t1 | 1024 | 1 | 6.206 | 16.6 | 0.77 |
| GRU_b1_t4 | 1 | 4 | 10.19 | 10.1 | 0.47 |
| GRU_b64_t4 | 64 | 4 | 0.672 | 153.2 | 7.11 |
| GRU_b1024_t4 | 1024 | 4 | 0.378 | 272.5 | 12.64 |
| neural ODE_b1_t4 | 1 | 4 | 690.1 | 0.1 | 0.01 |
| neural ODE_b64_t4 | 64 | 4 | 28.51 | 3.6 | 0.17 |
| neural ODE_b1024_t4 | 1024 | 4 | 4.783 | 21.5 | 1.00 |
| ensemble5_b1024_t4 | 1024 | 4 | 1.787 | 57.6 | 2.67 |

Reference check: dataset integrator (RK4, dt/5) vs scipy LSODA (rtol 1e-9) on 50 test traces: max |dV| = 1.58e-02 mV, RMSE = 3.43e-04 mV, mean Gamma = 1.000.

f-I curves: mean absolute rate error 7.41 Hz.

Total pipeline time: 206 s.
