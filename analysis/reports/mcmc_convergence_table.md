# MCMC Convergence Diagnostics Summary Table

Table of sampler health metrics across the 8 fitted Bayesian GAMM models in `fits/`:

|Model                                        | Max $\hat{R}$|$\hat{R} \le 1.01$ (%) | Min Bulk-ESS| Min Tail-ESS| Divergences| Min E-BFMI| Treedepth Setting| Max Treedepth Hits|
|:--------------------------------------------|-------------:|:----------------------|------------:|------------:|-----------:|----------:|-----------------:|------------------:|
|Tensor (k=10)                                |        1.0038|100.0%                 |         1479|         1531|           0|      0.801|                10|                  0|
|Smooth interaction (k=10)                    |        1.0059|100.0%                 |         1117|         2004|           0|      0.791|                10|                  0|
|Tensor &#124; Token Freq. (k=10)             |        1.0054|100.0%                 |         1253|         1695|           1|      0.799|                10|                  0|
|Tensor &#124; Token Freq. (k=4)              |        1.0049|100.0%                 |         1319|         1286|           1|      0.718|                10|                  0|
|Tensor (k=4)                                 |        1.0047|100.0%                 |         1117|         1030|           0|      0.720|                10|                  0|
|Smooth interaction (k=4)                     |        1.0107|99.8%                  |         1284|         1350|           2|      0.743|                10|                  0|
|Smooth interaction &#124; Token Freq. (k=10) |        1.0027|100.0%                 |         1015|         1233|           0|      0.751|                10|                  0|
|Smooth interaction &#124; Token Freq. (k=4)  |        1.0035|100.0%                 |         1212|         1276|           0|      0.734|                10|                  0|

> **Interpretation**:
> - 7 of 8 models satisfy $\hat{R} \le 1.01$ across all parameters. The largest value is 1.0107. Read the table before you use the affected models.
> - Bulk-ESS and Tail-ESS exceed the reliability threshold of 400 for 4 chains. The lowest values are 1015 (bulk) and 1030 (tail).
> - Divergent transitions range from 0 to 2 per model under `adapt_delta = 0.99`, out of 8,000 post-warmup draws.
> - Every chain-level E-BFMI value is at least 0.3.

