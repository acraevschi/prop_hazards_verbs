# MCMC Convergence Diagnostics Summary Table

Table of sampler health metrics across the 5 fitted Bayesian GAMM models in `fits/`:

|Model                            | Max $\hat{R}$|$\hat{R} \le 1.01$ (%) | Min Bulk-ESS| Min Tail-ESS| Divergences| Max Treedepth Hits|
|:--------------------------------|-------------:|:----------------------|------------:|------------:|-----------:|------------------:|
|Tensor (k=10)                    |        1.0056|100.0%                 |         1211|         2569|           0|                  0|
|Smooth interaction (k=10)        |        1.0071|100.0%                 |          777|         1666|           0|                  0|
|Tensor &#124; Token Freq. (k=10) |        1.0043|100.0%                 |         1078|          634|           0|               2000|
|Tensor (k=4)                     |        1.0039|100.0%                 |         1059|         1253|           9|                  0|
|Smooth interaction (k=4)         |        1.0086|100.0%                 |         1221|         1492|           1|                  0|

> **Interpretation**:
> - All 5 models satisfy $\hat{R} \le 1.01$ across 100% of estimated parameters (Vehtari et al., 2021). The largest value is 1.0086.
> - Bulk-ESS and Tail-ESS exceed the reliability threshold of 400 for 4 chains. The lowest values are 777 (bulk) and 634 (tail).
> - Divergent transitions range from 0 to 9 per model under `adapt_delta = 0.99`, out of 8,000 post-warmup draws.
> - Tensor | Token Freq. (k=10) hit the maximum treedepth of 10. This costs sampling efficiency, but it does not bias the posterior, and the $\hat{R}$ and ESS values for that model stay within the thresholds above.

