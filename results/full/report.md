# Results tables (full config)

## Architecture selection (validation split, single model each)

| model | params | train time (s) | mean Gamma | exact spike count | median RMSE (mV) | median sub-threshold RMSE (mV) | speed-up vs scipy |
|---|---|---|---|---|---|---|---|
| GRU | 203,269 | 1244 | 0.948 | 0.880 | 1.65 | 0.47 | 120x |
| neural ODE | 35,976 | 1306 | 0.949 | 0.953 | 1.56 | 0.28 | 16x |

Rule: candidates meeting the speed-up requirement (all, if none does) whose validation mean Gamma is within 0.01 of the best are equally accurate; among them the fastest is chosen (requirement: batched speed-up >= 100x vs scipy LSODA). Selected: **GRU**.

## Held-out accuracy of the twin (ensemble mean)

| split | traces | spiking | mean Gamma | median Gamma | exact spike count | spike-count R^2 | median RMSE (mV) | median sub-thr. RMSE (mV) | 1st-spike latency err (ms) |
|---|---|---|---|---|---|---|---|---|---|
| test_id | 300 | 0.85 | 0.934 | 1.000 | 0.877 | 0.911 | 1.70 | 0.48 | 0.02 |
| test_ood_param | 300 | 0.80 | 0.900 | 1.000 | 0.850 | 0.779 | 2.60 | 0.59 | 0.06 |
| test_ood_stim | 300 | 1.00 | 0.755 | 1.000 | 0.593 | -1.946 | 3.71 | 0.71 | 0.02 |

Single ensemble member (average over members), same splits:

| split | mean Gamma | exact spike count | median RMSE (mV) |
|---|---|---|---|
| test_id | 0.939 | 0.882 | 1.84 |
| test_ood_param | 0.899 | 0.904 | 2.69 |
| test_ood_stim | 0.804 | 0.667 | 3.91 |

Summary-feature errors (mean absolute error, ensemble mean trace):

| split | firing rate (Hz) | first-spike latency (ms) | mean sub-thr. V (mV) | peak V (mV) |
|---|---|---|---|---|
| test_id | 2.50 | 0.71 | 0.48 | 4.58 |
| test_ood_param | 4.03 | 1.94 | 1.05 | 6.40 |
| test_ood_stim | 17.23 | 0.09 | 1.42 | 2.10 |

Mean Gamma by stimulus family (* = never seen in training):

| noise | pulses | ramp | step | chirp* | strong_step* |
|---|---|---|---|---|---|
| 0.969 | 0.942 | 0.761 | 0.983 | 0.973 | 0.474 |

## Uncertainty

| split | median sigma (mV) | 2-sigma coverage | Spearman(sigma, RMSE) | flagged as extrapolation |
|---|---|---|---|---|
| test_id | 0.43 | 0.727 | 0.909 | 0.097 |
| test_ood_param | 0.61 | 0.712 | 0.894 | 0.243 |
| test_ood_stim | 0.87 | 0.731 | 0.861 | 0.357 |

- flag rate when sigma is combined with a parameter range check: test_id 0.097, test_ood_param 1.000, test_ood_stim 0.357
- flag threshold (95th pct of validation sigma): 1.23 mV
- AUROC ID vs extrapolated params: 0.639; ID vs novel stimuli: 0.822
- Spearman(sigma, RMSE), pooled test splits: 0.901
- Spearman(sigma, distance outside training box): 0.166
- mean Gamma flagged / unflagged: 0.510 / 0.970; median RMSE flagged / unflagged: 11.90 / 1.99 mV

## Speed (seconds per 100 ms trace, CPU)

| method | batch | threads | ms / trace | speed-up vs scipy | speed-up vs numpy RK4 |
|---|---|---|---|---|---|
| scipy_lsoda | 1 | 1 | 90.13 | 1.0 | 0.04 |
| numpy_rk4_batch | 1024 | 1 | 3.875 | 23.3 | 1.00 |
| GRU_b1_t1 | 1 | 1 | 11.22 | 8.0 | 0.35 |
| GRU_b64_t1 | 64 | 1 | 1.208 | 74.6 | 3.21 |
| GRU_b1024_t1 | 1024 | 1 | 1.764 | 51.1 | 2.20 |
| neural ODE_b1_t1 | 1 | 1 | 562.6 | 0.2 | 0.01 |
| neural ODE_b64_t1 | 64 | 1 | 19.51 | 4.6 | 0.20 |
| neural ODE_b1024_t1 | 1024 | 1 | 9.976 | 9.0 | 0.39 |
| ensemble5_b1024_t1 | 1024 | 1 | 9.169 | 9.8 | 0.42 |
| GRU_b1_t4 | 1 | 4 | 11.42 | 7.9 | 0.34 |
| GRU_b64_t4 | 64 | 4 | 0.93 | 97.0 | 4.17 |
| GRU_b1024_t4 | 1024 | 4 | 0.541 | 166.7 | 7.17 |
| neural ODE_b1_t4 | 1 | 4 | 617.7 | 0.1 | 0.01 |
| neural ODE_b64_t4 | 64 | 4 | 20.97 | 4.3 | 0.18 |
| neural ODE_b1024_t4 | 1024 | 4 | 4.149 | 21.7 | 0.93 |
| ensemble5_b1024_t4 | 1024 | 4 | 3.45 | 26.1 | 1.12 |

Reference check: dataset integrator (RK4, dt/5) vs scipy LSODA (rtol 1e-9) on 50 test traces: max |dV| = 1.58e-02 mV, RMSE = 3.43e-04 mV, mean Gamma = 1.000.

f-I curves: mean absolute rate error 7.04 Hz.

Total pipeline time: 2848 s.
