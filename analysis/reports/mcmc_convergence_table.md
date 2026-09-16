# MCMC Convergence Diagnostics Summary Table

Table of sampler health metrics across the 6 fitted Bayesian GAMM models in `fits/`:

|Model                            | Max $\hat{R}$|$\hat{R} \le 1.01$ (%) | Min Bulk-ESS| Min Tail-ESS| Divergences| Min E-BFMI| Treedepth Setting| Max Treedepth Hits|
|:--------------------------------|-------------:|:----------------------|------------:|------------:|-----------:|----------:|-----------------:|------------------:|
|Tensor (k=10)                    |        1.0028|100.0%                 |         1819|         1732|           0|      0.892|                10|                  0|
|Smooth interaction (k=10)        |        1.0055|100.0%                 |         1693|         2931|           1|      0.844|                10|                  0|
|Tensor &#124; Token Freq. (k=10) |        1.0028|100.0%                 |         1432|         2296|           0|      0.837|                10|                  0|
|Tensor &#124; Token Freq. (k=4)  |        1.0049|100.0%                 |         1496|         3542|           0|      0.775|                10|                  0|
|Tensor (k=4)                     |        1.0046|100.0%                 |         1187|         2214|           1|      0.730|                10|                  0|
|Smooth interaction (k=4)         |        1.0039|100.0%                 |         1446|         2659|           0|      0.786|                10|                  0|

> **Interpretation**:
> - All 6 models satisfy $\hat{R} \le 1.01$ across 100% of estimated parameters (Vehtari et al., 2021). The largest value is 1.0055.
> - Bulk-ESS and Tail-ESS exceed the reliability threshold of 400 for 4 chains. The lowest values are 1187 (bulk) and 1732 (tail).
> - Divergent transitions range from 0 to 1 per model under `adapt_delta = 0.99`, out of 8,000 post-warmup draws.
> - Every chain-level E-BFMI value is at least 0.3.

