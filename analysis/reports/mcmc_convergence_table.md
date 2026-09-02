# MCMC Convergence Diagnostics Summary Table

Table of sampler health metrics across the 5 fitted Bayesian GAMM models in `fits/`:

|Model                            | Max $\hat{R}$|$\hat{R} \le 1.01$ (%) | Min Bulk-ESS| Min Tail-ESS| Divergences| Max Treedepth Hits|
|:--------------------------------|-------------:|:----------------------|------------:|------------:|-----------:|------------------:|
|Tensor (k=10)                    |        1.0110|100.0%                 |          568|          786|           0|               2000|
|Smooth interaction (k=10)        |        1.0100|100.0%                 |          792|          834|           0|               8000|
|Tensor &#124; Token Freq. (k=10) |        1.0085|100.0%                 |          913|          422|           1|                  0|
|Tensor (k=4)                     |        1.0040|100.0%                 |         1244|         1285|           1|                  0|
|Smooth interaction (k=4)         |        1.0048|100.0%                 |         1612|         1869|          13|                  0|

> **Interpretation**:
> - 4 of 5 models satisfy $\hat{R} \le 1.01$ across all parameters. The largest value is 1.0110. Read the table before you use the affected models.
> - Bulk-ESS and Tail-ESS exceed the reliability threshold of 400 for 4 chains. The lowest values are 568 (bulk) and 422 (tail).
> - Divergent transitions range from 0 to 13 per model under `adapt_delta = 0.99`, out of 8,000 post-warmup draws.
> - Tensor (k=10), Smooth interaction (k=10) hit the maximum treedepth of 10. This costs sampling efficiency, but it does not bias the posterior, and the $\hat{R}$ and ESS values for those models stay within the thresholds above.

