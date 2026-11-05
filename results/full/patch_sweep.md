# Patch sweep (full config)

CPU threads: 4; GPU: none available; scipy LSODA: 87.171 ms/trace; 60 epochs per model.

| step (ms) | GRU steps / trace | val Γ | test Γ | test exact count | test median RMSE (mV) | extrap. params Γ | novel stimuli Γ | cpu_b1 ms/trace (speed-up) | cpu_b1024 ms/trace (speed-up) |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 200 | 0.948 | 0.927 | 0.89 | 1.80 | 0.895 | 0.795 | 11.75 (7×) | 0.699 (125×) |
| 1 | 100 | 0.967 | 0.942 | 0.89 | 1.40 | 0.905 | 0.771 | 9.877 (9×) | 0.3428 (254×) |
| 2 | 50 | 0.957 | 0.933 | 0.87 | 1.90 | 0.921 | 0.785 | 3.733 (23×) | 0.1364 (639×) |
| 5 | 20 | 0.918 | 0.905 | 0.78 | 3.49 | 0.886 | 0.730 | 1.842 (47×) | 0.04822 (1,808×) |
