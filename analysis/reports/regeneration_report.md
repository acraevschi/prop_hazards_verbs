# Task 2 regeneration report

Generated 2026-09-16 from the Task 1 identity and lemma-join repair at
`a452790`. No manuscript prose was edited.

## Result

The repaired pipeline was regenerated from corpus extraction through reporting.
All six active fits were newly sampled from the same analysis dataset; the fit
sidecars record `sampled_fresh = TRUE`, the dataset MD5, sampling times, formula,
and runtime settings. Every formula contains `(1 | document_id)`. The model data
contain 226 corpus-qualified source documents: 75 ENHG and 151 MHG.

The final likelihood unit is deliberately
`distinct(document_id, lemma_id, std_infl, has_levelled)`. Thus an all-preserved
cell contributes one `0`, an all-levelled cell one `1`, and a mixed cell one of
each. Repetition within an outcome state does not weight the likelihood. The
7,510 Bernoulli observations come from 7,422 document–lemma–slot cells and retain
counts for 38,291 underlying vowel outcomes as audit fields only. The separate
vowel/consonant analysis remains paired on token-level `observation_id`.

The principal conclusion is not strengthened by the repair. In the leading
model, the bipartite-vowel coefficient is -0.186 with 95% credible interval
[-1.353, 0.990], posterior probability 0.621 that it is negative, and evidence
ratio 1.64. The interval comfortably spans zero.

## Commands and runtime

The dependency order and commands used were:

```bash
python data/extract_mhg_data.py
python data/extract_enhg_data.py
python data/lemmas/enhg_mhg_mapping.py
python data/normalize_data.py
python data/extract_nhg_preterites.py
python data/build_nhg_targets.py
python data/corpus_approach_coding.py

Rscript analysis/run_brms.R --prepare-only --threads 4
python analysis/attrition_diagnostics.py
python analysis/marking_type_summary.py
python analysis/consonant_analysis.py
python analysis/target_sensitivity.py
python -m pytest -q
Rscript analysis/run_brms.R --dry-run --chains 4 --cores 4 --threads 4

Rscript analysis/run_brms.R --chains 4 --cores 4 --threads 4

python analysis/attrition_diagnostics.py
python analysis/marking_type_summary.py --sensitivity
python analysis/consonant_analysis.py
python analysis/target_sensitivity.py
python -m pytest -q
Rscript analysis/mcmc_convergence.R
Rscript -e "rmarkdown::render('analysis/analyze_models.Rmd')"
```

The production command was stopped after Model 4 at the user's request and run
again without `--overwrite`. The second invocation skipped Models 1–4 and sampled
Models 5–6. Defaults resolved to 4 chains, 4,000 iterations, 2,000 warmup,
seed 97, `adapt_delta = 0.99`, and maximum tree depth 10. Four parallel chains
with four within-chain threads used up to 16 CPU threads. The pinned, cached
UniMorph checkout at revision `d226d21` was reused and its expected checksum
validated.

Runtime: Python 3.13.9; R 4.4.3; brms 2.23.0; cmdstanr 0.8.0;
CmdStan 2.38.0; rstan 2.32.7; loo 2.9.0; posterior 1.6.1;
rmarkdown 2.30; dplyr 1.2.0; tidyr 1.3.2; marginaleffects 0.32.0;
macOS 26.5.2 arm64.

## Fresh-fit provenance

The final dataset SHA-256 is
`caf0c62f6b76fa923b0440353a57a570f37aa6064fc12990ff5cba679adf4341`.
Its MD5, repeated in all six sidecars, is
`b5b7bd22b4f2d12c676807be4bc0365b`.

| fit | sampled (CEST) | SHA-256 |
| :--- | :--- | :--- |
| `base_fit_marking_type.rds` | 08:03–08:16 | `770a77c7287d5bb4bea7f74dd7c3398758231a5a94b00e3f8f53d15fa8f91cae` |
| `base_fit_marking_type_k10.rds` | 08:16–08:33 | `6073cd2a1275118252896fb4c9cbb1abd736a2282b79269633d8445603c1a4f8` |
| `tensor_fit_marking_type_k10.rds` | 08:33–09:01 | `74eeb9e534e5d05e8c34bee1a65a89b31e33db98f9258c68621993b12b17f088` |
| `tensor_fit_marking_type_k4.rds` | 09:01–09:14 | `5b1f49226b6bb2304a37ad0bd65caff9716f0674b99fcc88d5f140272ad4b8d8` |
| `tensor_fit_marking_type_k10_token.rds` | 11:12–11:40 | `d5cece96b90f46c839d00073accd09009713f213e098ca02455b963b1acd3c04` |
| `tensor_fit_marking_type_k4_token.rds` | 11:40–11:54 | `3de863967dde3d37281597d438932861b7300d8d6a02a580b73e52b256307882` |

The pre-Task-2 fits are preserved under
`runs/pre-task2-id-collapsed-20260915/fits/`. An interim binomial-cell
interpretation is preserved under `runs/task2-binomial-cell-20260916/fits/` but
is not an active fit and supplies no reported result.

## Stage-by-stage counts

`observations` uses the unit named in each row; `source tokens` prevents the
collapsed model rows from being mistaken for token counts.

| stage | relevant denominator | observations | source tokens | surface lemmas | lemma families | documents | unit or gate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| MHG extraction | 164,187 | 164,187 | 164,187 | 2,945 | — | 405 | extracted token |
| ENHG extraction | 166,800 | 166,800 | 166,800 | 1,754 | — | 190 | extracted token |
| combined join | 330,987 | 330,987 | 330,987 | 4,671 | 455 | 595 | mapped and unmapped retained |
| mapped join | 330,987 | 281,458 | 281,458 | 3,406 | 455 | 594 | non-missing corpus-aware family |
| frequency eligible | 281,458 | 280,870 | 280,870 | 3,175 | 291 | 594 | family frequency > 10 |
| normalized | 280,870 | 142,045 | 142,045 | 2,608 | 290 | 234 | date, variety, and principal part present |
| coding eligible | 142,045 | 108,389 | 108,389 | 2,335 | 282 | 234 | eligible strong-verb class |
| coded past tokens | 108,389 | 47,248 | 47,248 | 1,587 | 233 | 226 | PastSg or PastPl emitted by coder |
| vowel model | 47,248 | 7,510 | 38,291 | 124 | 124 | 226 | distinct document–family–slot–outcome |

The join tripwire reconciles as 330,987 = 281,458 mapped + 49,529 unmapped.
By corpus, the model represents 8,737 ENHG vowel tokens as 1,526 observations
from 75 documents, and 29,554 MHG tokens as 5,984 observations from 151
documents.

Frequency attrition is 588 tokens. From the 280,870 frequency-eligible tokens,
independent missing-field counts are 127,410 missing date, 9,937 missing variety,
and 8,276 missing principal part; they overlap. Applying exclusive precedence in
that order gives 127,410, 7,612, and 3,803 exclusions, respectively, and leaves
142,045 normalized tokens.

The baseline uses exactly `Pres`, `PastSg`, and `PastPl` at `date <= 1200`.
It contains 954 supported cells, including 160 one-token cells. Participles are
excluded because neither production contrast uses them.

Target-source counts use all 531 normalized lemma–variety groups as denominator.
Present targets resolve for 406 (all modern) and are missing for 125. Past targets
resolve for 400: 397 modern, 6 corpus fallback, and 3 unresolved among otherwise
present target rows; 125 groups have no target row.

Production coding excludes 528 past and 15 present comparisons under the
protected regular-sound-change rule. The unprotected counterfactual gives 2,508,
not the obsolete 9,079 figure.

## Outcomes and sensitivities

At token level there are 35,413 unipartite vowel observations (351 levelled),
2,878 bipartite vowel observations (17 levelled), and 1,212 bipartite consonant
observations (68 levelled). After the explicit GAMM collapse there are 6,707
unipartite rows (188 levelled) and 803 bipartite rows (12 levelled).

The 12 model-unit bipartite events occur in five lemma families: `ziehen`,
`lîden`, and `lîhen` contribute three each, `snîden` two, and `zîhen` one.
Forcing `lîhen` to be bipartite in both varieties changes the bipartite model
support from 803/12 events to 821/16. Requiring at least two agreeing tokens for
every anchor changes it to 781/8. Flipping all rejected modern variants leaves
the 7,510 model observations unchanged and removes two unipartite events.

The matched channel analysis increases from 711 old document/slot matches to
1,083 collision-free token pairs across nine verbs. The old discordance was
48 consonant-only versus 5 vowel-only (53 total; exact p = 7.08e-10;
lemma-clustered 95% interval 60.0%–100.0%). The repaired analysis has 51 versus
10 (61 total; exact p = 9.624e-08; clustered interval 36.4%–100.0%). The wider,
lower interval must be reported rather than only the exact test.

Both target alternatives have identical outcome labels among the 38,291 vowel
observations codable under both definitions, but 8,957 of 47,248 coded rows are
not vowel-codable under either. Direct target agreement is a separate audit: the
strict late-corpus definition resolves 398 past-target groups in common with
production, while production alone resolves two more; 131 of 531 groups are
unresolved under at least one definition. These denominators prevent the shared
subset's 100% agreement from being described as universal target identity.

## Old versus new Bayesian results

The old fits used 17,467 observations, 124 lemmas, and an overloaded `id` random
intercept. The new fits use 7,510 explicitly deduplicated observations, 124 lemma
families, and 226 real source-document intercept levels. Raw ELPD values are not
comparable across datasets of different size; ranks and within-run differences
are the useful comparison.

| model | old rank | new rank | old ELPD | new ELPD | old bipartite coefficient [95% CrI] | new coefficient [95% CrI] |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| Tensor (k=10) Token | 1 | 1 | -980.9 | -651.1 | -0.280 [-1.563, 1.010] | -0.186 [-1.353, 0.990] |
| Tensor (k=10) | 3 | 2 | -997.4 | -651.6 | -0.463 [-1.705, 0.770] | -0.199 [-1.389, 0.982] |
| Base (k=10) | 2 | 3 | -991.4 | -652.1 | -0.464 [-1.708, 0.775] | -0.189 [-1.378, 0.985] |
| Base (k=4) | 5 | 4 | -1033.1 | -657.2 | -0.561 [-1.782, 0.618] | -0.184 [-1.346, 0.959] |
| Tensor (k=4) Token | 4 | 5 | -1021.7 | -657.7 | -0.277 [-1.505, 0.928] | -0.181 [-1.327, 0.959] |
| Tensor (k=4) | 6 | 6 | -1033.7 | -658.4 | -0.544 [-1.767, 0.660] | -0.212 [-1.348, 0.929] |

The leading model remains Tensor (k=10) Token. The next two are statistically
close: Tensor (k=10) has ELPD difference -0.53 (SE 2.26), and Base (k=10)
-0.95 (SE 2.60). The repaired coefficients are uniformly nearer zero, and every
95% credible interval spans zero.

## Tests and diagnostics

- Python: 72 tests passed.
- The pre-fit check independently reproduced 7,510 rows: 6,707 unipartite
  (188 levelled) and 803 bipartite (12 levelled).
- `--dry-run` constructed and validated all six Stan programs before sampling.
- Across fits, maximum R-hat ranges from 1.0028 to 1.0055; every reported
  parameter has R-hat <= 1.01. The global minimum bulk ESS is 1,187 and minimum
  tail ESS is 1,732. Minimum chain E-BFMI is 0.730.
- Smooth interaction k=10 and Tensor k=4 each have one divergent transition;
  the other four fits have none. No fit hit the configured maximum tree depth.
- PSIS-LOO has no observations with Pareto-k >= 0.7. Per-model maxima range
  from 0.568 to 0.668.
- Stan compilation emitted linker warnings that bundled CmdStan/Sundials object
  files target macOS 16.0 while the model link target is 11.0. Compilation and
  all chains completed; this is a toolchain warning rather than a sampling
  diagnostic.
- The first report render found a stale seven-column display label for the new
  nine-column convergence table. The label list was corrected; the final HTML
  render completed successfully.

## Source of truth for manuscript revision

| number or claim | generated source |
| :--- | :--- |
| stage, token, lemma-family, and document counts | `analysis/reports/stage_counts.csv` and `corpus_stage_counts.csv` |
| mapped and unmapped join counts | `analysis/reports/lemma_join_counts.csv` |
| frequency and metadata attrition | `analysis/reports/normalization_exclusions.csv` and `normalization_exclusions_exclusive.csv` |
| baseline size and support | `analysis/reports/baseline_anchor_support.csv` and `baseline_support_distribution.csv` |
| target sources and missing denominator | `analysis/reports/target_source_counts.csv` |
| production sound-change exclusions | `analysis/reports/attrition_report.md` |
| token-level vowel/consonant outcomes | `analysis/reports/channel_outcomes.csv` |
| final GAMM observation counts | `analysis/data_for_analysis.csv`, `model_observation_summary.csv`, and `model_fit_support.csv` |
| predictor distributions | `analysis/reports/model_predictor_distribution.csv` |
| marking-type concentration and `lîhen` | `model_event_concentration.csv`, `marking_type_lemma_events.csv`, and `lihen_sensitivity.csv` |
| rejected-modern-variant sensitivity | `analysis/reports/rejected_variant_sensitivity.csv` |
| target-definition agreement and exclusions | `target_sensitivity_summary.csv` and `target_sensitivity_targets.csv` |
| matched channel test and clustered interval | `consonant_analysis_report.md` and `consonant_paired_discordance.csv` |
| LOO ordering and Pareto-k | `model_comparison.csv` and `psis_loo_diagnostics.csv` |
| coefficients and credible intervals | `posterior_fixed_effects.csv` |
| directional evidence ratio | `evidence_ratios.csv` |
| R-hat, ESS, divergences, E-BFMI, tree depth | `mcmc_convergence.csv` |
| fit hashes, timestamps, data hash, settings | `fits/*.provenance.csv` and this report |
| rendered complete model report and figures | `analysis/analyze_models.html` and `figures/*.pdf` |

