# Results tables (small config)

## Architecture selection (validation split, single model each)

| model | params | train time (s) | mean Gamma | exact spike count | median RMSE (mV) | median sub-threshold RMSE (mV) | speed-up vs scipy |
|---|---|---|---|---|---|---|---|
| GRU | 52,485 | 82 | 0.856 | 0.720 | 3.42 | 1.15 | 174x |
| neural ODE | 5,640 | 78 | 0.702 | 0.580 | 11.05 | 2.73 | 13x |

Rule: candidates meeting the speed-up requirement (all, if none does) whose validation mean Gamma is within 0.01 of the best are equally accurate; among them the fastest is chosen (requirement: batched speed-up >= 100x vs scipy LSODA). Selected: **GRU**.

## Held-out accuracy of the twin (ensemble mean)

| split | traces | spiking | mean Gamma | median Gamma | exact spike count | spike-count R^2 | median RMSE (mV) | median sub-thr. RMSE (mV) | 1st-spike latency err (ms) |
|---|---|---|---|---|---|---|---|---|---|
| test_id | 100 | 0.79 | 0.890 | 1.000 | 0.700 | 0.725 | 3.91 | 1.06 | 0.05 |
| test_ood_param | 100 | 0.76 | 0.765 | 1.000 | 0.640 | 0.328 | 5.12 | 1.55 | 0.08 |
| test_ood_stim | 100 | 1.00 | 0.568 | 0.458 | 0.290 | -3.140 | 10.17 | 4.61 | 0.05 |

Single ensemble member (average over members), same splits:

| split | mean Gamma | exact spike count | median RMSE (mV) |
|---|---|---|---|
| test_id | 0.896 | 0.737 | 4.06 |
| test_ood_param | 0.799 | 0.690 | 5.21 |
| test_ood_stim | 0.574 | 0.307 | 10.46 |

Summary-feature errors (mean absolute error, ensemble mean trace):

| split | firing rate (Hz) | first-spike latency (ms) | mean sub-thr. V (mV) | peak V (mV) |
|---|---|---|---|---|
| test_id | 6.10 | 0.76 | 1.16 | 6.51 |
| test_ood_param | 11.60 | 1.02 | 2.38 | 13.25 |
| test_ood_stim | 28.00 | 0.27 | 3.57 | 4.75 |

Mean Gamma by stimulus family (* = never seen in training):

| noise | pulses | ramp | step | chirp* | strong_step* |
|---|---|---|---|---|---|
| 0.889 | 0.932 | 0.857 | 0.872 | 0.865 | 0.283 |

## Uncertainty

| split | median sigma (mV) | 2-sigma coverage | Spearman(sigma, RMSE) | flagged as extrapolation |
|---|---|---|---|---|
| test_id | 0.75 | 0.613 | 0.813 | 0.030 |
| test_ood_param | 1.07 | 0.594 | 0.900 | 0.190 |
| test_ood_stim | 2.16 | 0.596 | 0.808 | 0.390 |

- flag rate when sigma is combined with a parameter range check: test_id 0.030, test_ood_param 1.000, test_ood_stim 0.390
- flag threshold (95th pct of validation sigma): 2.89 mV
- AUROC ID vs extrapolated params: 0.625; ID vs novel stimuli: 0.884
- Spearman(sigma, RMSE), pooled test splits: 0.881
- Spearman(sigma, distance outside training box): -0.098
- mean Gamma flagged / unflagged: 0.323 / 0.847; median RMSE flagged / unflagged: 15.23 / 4.38 mV

## Speed (seconds per 100 ms trace, CPU)

| method | batch | threads | ms / trace | speed-up vs scipy | speed-up vs numpy RK4 |
|---|---|---|---|---|---|
| scipy_lsoda | 1 | 1 | 123.7 | 1.0 | 0.08 |
| numpy_rk4_batch | 256 | 1 | 10.03 | 12.3 | 1.00 |
| GRU_b1_t1 | 1 | 1 | 9.038 | 13.7 | 1.11 |
| GRU_b256_t1 | 256 | 1 | 0.685 | 180.5 | 14.63 |
| neural ODE_b1_t1 | 1 | 1 | 435 | 0.3 | 0.02 |
| neural ODE_b256_t1 | 256 | 1 | 5.261 | 23.5 | 1.91 |
| ensemble3_b256_t1 | 256 | 1 | 2.148 | 57.6 | 4.67 |
| GRU_b1_t4 | 1 | 4 | 9.189 | 13.5 | 1.09 |
| GRU_b256_t4 | 256 | 4 | 0.522 | 237.0 | 19.22 |
| neural ODE_b1_t4 | 1 | 4 | 433.9 | 0.3 | 0.02 |
| neural ODE_b256_t4 | 256 | 4 | 5.669 | 21.8 | 1.77 |
| ensemble3_b256_t4 | 256 | 4 | 1.377 | 89.8 | 7.28 |

Reference check: dataset integrator (RK4, dt/5) vs scipy LSODA (rtol 1e-9) on 20 test traces: max |dV| = 7.92e-03 mV, RMSE = 1.56e-04 mV, mean Gamma = 1.000.

f-I curves: mean absolute rate error 24.07 Hz.

Total pipeline time: 383 s.
