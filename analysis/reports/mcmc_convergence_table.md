# MCMC Convergence Diagnostics Summary Table

Table of sampler health metrics across the 5 fitted Bayesian GAMM models in `fits/`:

|Model                            | Max $\hat{R}$|$\hat{R} \le 1.01$ (%) | Min Bulk-ESS| Min Tail-ESS| Divergences| Max Treedepth Hits|
|:--------------------------------|-------------:|:----------------------|------------:|------------:|-----------:|------------------:|
|Tensor (k=10)                    |        1.0097|100.0%                 |          577|         1022|           0|                  0|
|Smooth interaction (k=10)        |        1.0051|100.0%                 |         1238|         1098|           3|                  0|
|Tensor &#124; Token Freq. (k=10) |        1.0066|100.0%                 |         1123|         1359|           0|                  0|
|Tensor &#124; Token Freq. (k=4)  |        1.0057|100.0%                 |          724|         1374|           0|                  0|
|Tensor (k=4)                     |        1.0076|100.0%                 |          804|         2231|           0|                  0|
|Smooth interaction (k=4)         |        1.0070|100.0%                 |          570|         1425|           0|                  0|

> **Interpretation**:
> - All 6 models satisfy $\hat{R} \le 1.01$ across 100% of estimated parameters (Vehtari et al., 2021). The largest value is 1.0097.
> - Bulk-ESS and Tail-ESS exceed the reliability threshold of 400 for 4 chains. The lowest values are 570 (bulk) and 1022 (tail).
> - Divergent transitions range from 0 to 3 per model under `adapt_delta = 0.99`, out of 8,000 post-warmup draws.

