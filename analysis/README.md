# Statistical Analysis & Diagnostics

This directory contains the statistical modeling, MCMC convergence diagnostics, data attrition audits, and sensitivity analyses for testing **Hermann Paul’s Principle** in High German strong verbs.

---

## 📂 Directory Layout

```
analysis/
├── README.md                                <- This documentation file
├── data_for_analysis.csv                    <- Prepared modeling dataset (13,184 rows, 88 unique lemmas)
│
├── 🧠 Core Bayesian Modeling Pipeline
│   ├── run_brms.R                           <- Fits 5 Bayesian GAMM models via brms / Stan
│   ├── analyze_models.Rmd                   <- LOO-CV model comparison, hypothesis testing, figure exports
│   └── analyze_models.html                  <- Rendered R Markdown analysis report
│
├── 🩺 Post-Fit Diagnostics & Sampler Health
│   └── mcmc_convergence.R                  <- Computes R-hat, Bulk/Tail ESS, divergences across fits/
│
├── 🔍 Pre-Modeling Data & Target State Audits
│   ├── attrition_diagnostics.py             <- Tracks lemma/token retention and sound change vs. leveling
│   ├── target_sensitivity.py                <- Double robustness check for late ENHG target definitions
│   └── check_alternation_patterns.ipynb     <- Interactive visual inspection of alternation patterns
│
└── 📊 reports/                              <- Audit summaries & diagnostic reports (.md, .csv)
    ├── attrition_report.md                  <- Full funnel report on lemma/token attrition
    ├── attrition_summary.csv                <- Key attrition metrics table
    ├── target_sensitivity_report.md         <- Report on target state stability & label concordance
    ├── target_sensitivity_summary.csv       <- Quantitative concordance table (100% agreement)
    ├── mcmc_convergence_table.md           <- Sampler convergence markdown summary table
    └── mcmc_convergence.csv                <- Machine-readable MCMC sampler diagnostics
```

---

## 📑 Detailed Script Catalog

### 1. Core Bayesian Modeling

* **`run_brms.R`**:
  - **Purpose**: Prepares model variables and fits 5 Bayesian Generalized Additive Mixed Models (GAMMs) using `brms` and Stan.
  - **Models Fitted**:
    1. `base_fit_marking_type` (Smooth Interaction GAMM, $k=4$, appendix baseline)
    2. `base_fit_marking_type_k10` (Smooth Interaction GAMM, $k=10$)
    3. `tensor_fit_marking_type_k10` (**Primary Model in Paper**, Tensor Product $t_2(\text{date}, \log(\text{freq}))$, $k=10$)
    4. `tensor_fit_marking_type_k4` (Tensor Product GAMM, $k=4$, sensitivity check on basis dimension)
    5. `tensor_fit_marking_type_k10_token` (Tensor Product with token frequency, sensitivity check on frequency definition)
  - **Key Predictors**:
    - `marking_type`: 3-level factor (`vowel_unipartite`, `vowel_bipartite`, `consonant_bipartite`) to avoid structural collinearity.
    - Smooth temporal trends: $s(\text{date}, k)$ and $s(\text{date}, \text{by}=\text{marking\_type}, k)$.
    - Alternation controls: `has_alt_pres`, `log_alt_pres_freq`, `has_alt_past`, `log_alt_past_freq`.
    - Random effects: `(1 | variety) + s(date, by = variety, k)`, `(1 | lemma_std)`, `(1 | id)`.
  - **LOO-CV**: Each fit stores Leave-One-Out Cross-Validation in `fit$criteria$loo`. This is the PSIS approximation from `loo()`. Read the Pareto-k diagnostics before you trust the model comparison.
  - **Outputs**: Serialized `.rds` model objects in `fits/` and prepared modeling data in `analysis/data_for_analysis.csv`.

* **`analyze_models.Rmd`**:
  - **Purpose**: Comprehensive post-processing, Leave-One-Out Cross-Validation (PSIS-LOO), MCMC convergence tables, hypothesis testing (evidence ratios), and publication figure generation.
  - **Outputs**: Publication-ready PDF figures in `figures/` and rendered HTML report `analyze_models.html`.

---

### 2. Sampler Convergence & Health Diagnostics

* **`mcmc_convergence.R`**:
  - **Purpose**: Audits Stan MCMC health across all 5 fitted models in `fits/` to guarantee reliable posterior exploration.
  - **Metrics**: Max $\hat{R}$, percentage of parameters with $\hat{R} \le 1.01$, minimum Bulk-ESS, minimum Tail-ESS, divergent transitions, and maximum treedepth hits.
  - **Outputs**: `reports/mcmc_convergence_table.md` and `reports/mcmc_convergence.csv`.

---

### 3. Pre-Modeling Data & Target State Audits (Double Robustness)

* **`attrition_diagnostics.py`**:
  - **Purpose**: Audits the longitudinal data pipeline from raw corpus texts (~1050–1650 CE) to the final GAMM dataset.
  - **Tracks**:
    - *Lemma Retention*: Traces lemmas through DSU unification, frequency filtering ($> 10$), pre-1200 start-state anchoring, and final GAMM inclusion.
    - *Token Retention*: Quantifies tokens dropped due to unanchored baselines vs. retained past-tense tokens.
    - *Sound Change vs. Leveling Filter*: Evaluates the regular sound-change filter (`data/vowel_changes.csv`), reporting how many transitions represent regular dialect sound changes (e.g. Upper German *uo* > *u*, Central German *î* > *ei*) rather than analogical leveling.
  - **Outputs**: `reports/attrition_report.md` and `reports/attrition_summary.csv`.

* **`target_sensitivity.py`**:
  - **Purpose**: Double robustness check verifying that defining each verb's teleological target state (the morphological endpoint by the end of ENHG) via `max(date)` per lemma is not biased by varying document survival dates.
  - **Regimes Evaluated**:
    1. *Baseline Target*: Modal vowel and coda at $\max(\text{date})$ per lemma.
    2. *Variant 1 (Strict Late Subset)*: Only lemmas attested in texts dated $\ge 1500$.
    3. *Variant 2 (Hybrid Fallback)*: Prioritizes texts dated $\ge 1500$ when available, falling back to $\max(\text{date})$ for lemmas whose records cease earlier.
  - **Result**: Confirms **100.00% label concordance** across all past-tense tokens (zero classification flips).
  - **Outputs**: `reports/target_sensitivity_report.md` and `reports/target_sensitivity_summary.csv`.

* **`check_alternation_patterns.ipynb`**:
  - Interactive notebook for exploring distribution of alternation types and validating empirical patterns across verb classes.

---

## 🚀 Execution & Replication Workflow

To reproduce the analysis and audit reports from scratch:

```bash
# 1. Run Pre-Modeling Diagnostics & Double Robustness Checks
python analysis/attrition_diagnostics.py
python analysis/target_sensitivity.py

# 2. Fit Bayesian GAMM Models (4 chains x 2 threads = 8 total CPU threads)
Rscript analysis/run_brms.R --chains 4 --iter 4000 --cores 4 --threads 2 --seed 97

# (Optional) Force re-estimation of models that are already cached in fits/:
# Rscript analysis/run_brms.R --overwrite

# 3. Generate MCMC Convergence Summary Table
Rscript analysis/mcmc_convergence.R

# 4. Render LOO-CV Comparison, Hypothesis Tests & Publication Figures
Rscript -e "rmarkdown::render('analysis/analyze_models.Rmd')"
```
