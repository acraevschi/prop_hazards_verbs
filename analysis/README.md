# Statistical Analysis & Diagnostics

This directory contains the statistical modeling, MCMC convergence diagnostics, data attrition audits, and sensitivity analyses for testing **Hermann Paul’s Principle** in High German strong verbs.

---

## 📂 Directory Layout

```
analysis/
├── README.md                                <- This documentation file
├── data_for_analysis.csv                    <- Prepared vowel-only modeling dataset (17,467 rows, 124 unique lemmas)
│
├── 🧠 Core Bayesian Modeling Pipeline
│   ├── run_brms.R                           <- Fits 6 Bayesian GAMM models via brms / Stan (Option A: Vowel-Only)
│   ├── analyze_models.Rmd                   <- LOO-CV model comparison, hypothesis testing, figure exports
│   └── analyze_models.html                  <- Rendered R Markdown analysis report
│
├── 🔬 Consonant Channel & Mechanism Audits
│   ├── consonant_analysis.py                <- Consonant channel audit + within-cell paired channel test
│   ├── marking_type_summary.py              <- Fast reshape & marking type breakdown straight from coded data
│   └── check_alternation_patterns.ipynb     <- Interactive visual inspection of alternation patterns
│
├── 🩺 Post-Fit Diagnostics & Sampler Health
│   └── mcmc_convergence.R                  <- Computes R-hat, Bulk/Tail ESS, divergences across fits/
│
├── 🔍 Pre-Modeling Data & Target State Audits
│   ├── attrition_diagnostics.py             <- Tracks lemma/token retention and sound change vs. leveling
│   └── target_sensitivity.py                <- Double robustness check for late ENHG target definitions
│
└── 📊 reports/                              <- Audit summaries & diagnostic reports (.md, .csv)
    ├── consonant_analysis_report.md         <- Comprehensive report on consonant channel leveling
    ├── consonant_summary.csv                <- Mechanism summary (Morphological GW vs Orthographic)
    ├── consonant_lemma_breakdown.csv        <- Per-lemma consonant leveling counts and rates
    ├── attrition_report.md                  <- Full funnel report on lemma/token attrition
    ├── attrition_summary.csv                <- Key attrition metrics table
    ├── target_sensitivity_report.md         <- Report on target state stability & label concordance
    ├── target_sensitivity_summary.csv       <- Quantitative concordance table (100% agreement)
    ├── mcmc_convergence_table.md           <- Sampler convergence markdown summary table
    └── mcmc_convergence.csv                <- Machine-readable MCMC sampler diagnostics
```

---

## 📑 Detailed Script Catalog

### 1. Core Bayesian Modeling (Option A: Vowel-Only Model)

* **`run_brms.R`**:
  - **Purpose**: Prepares model variables and fits 6 Bayesian Generalized Additive Mixed Models (GAMMs) using `brms` and Stan on the **vowel-only** dataset.
  - **Factor Specification**:
    - Consonant observations are filtered out to eliminate confounding from orthographic coda devoicing (*Auslautverhärtung*).
    - `marking_type` is parameterized with `vowel_unipartite` explicitly as the reference factor level (baseline $\beta_0$), so the `vowel_bipartite` parameter measures the treatment contrast directly.
  - **Models Fitted**:
    1. `tensor_fit_marking_type_k10_token` (**Primary Model in Paper**, Tensor Product $t_2(\text{date}, \log(\text{token\_freq}))$, $k=10$, best predictive performance $\Delta\text{elpd}=0.0$)
    2. `tensor_fit_marking_type_k4_token` (Tensor Product GAMM with token frequency, $k=4$, dimension sensitivity check)
    3. `tensor_fit_marking_type_k10` (Tensor Product GAMM with lemma frequency, $k=10$, frequency operationalization check)
    4. `tensor_fit_marking_type_k4` (Tensor Product GAMM with lemma frequency, $k=4$, sensitivity check on basis dimension)
    5. `base_fit_marking_type_k10` (Smooth Interaction GAMM with lemma frequency, $k=10$)
    6. `base_fit_marking_type` (Smooth Interaction GAMM with lemma frequency, $k=4$, appendix baseline)
  - **CLI Flags**: Supports `--test` (fast test fit) and `--dry-run` (validates formulas and Stan code without sampling) alongside `--chains`, `--iter`, `--cores`, `--threads`, `--backend`, and `--overwrite`.
  - **Outputs**: Serialized `.rds` model objects in `fits/` and prepared modeling data in `analysis/data_for_analysis.csv`.

* **`analyze_models.Rmd`**:
  - **Purpose**: Comprehensive post-processing, Leave-One-Out Cross-Validation (PSIS-LOO), MCMC convergence tables, hypothesis testing (`marking_typevowel_bipartite < 0`), and publication figure generation.
  - **Outputs**: Publication-ready PDF figures in `figures/` (`fixed_effects.pdf`, `bi-uni_diff.pdf`, `leveling_trajectories.pdf`, etc.) and rendered HTML report `analyze_models.html`.

---

### 2. Dedicated Consonant Analysis

* **`consonant_analysis.py`**:
  - **Purpose**: Standalone module and CLI tool auditing the consonant channel, whose leveling rate (7.35%) is well above both bipartite vowels (0.80%) and unipartite vowels (2.04%).
  - **Category is derived from the paradigm, not hand-listed**: `classify_consonant_lemma()` reads the same anchors and the same `diff_cons_*` flags, through the same clauses, that `step_2_establish_baseline` used to admit the paradigm. The two therefore cannot drift apart. Because the upstream shape test already rejects *Auslautverhärtung* (*scheiden* d ~ t ~ d), **every paradigm here is grammatischer Wechsel by construction**; the devoicing category is empty and is printed as a tripwire on the rule, not as a finding about the language.
  - **Two designs, kept apart**:
    - *Sections 1–4* report unpaired rates and odds ratios. These are **descriptive only** — the consonant rows and the vowel-bipartite rows are largely the same cells, so Fisher's exact test understates the uncertainty badly.
    - *Section 5* is the design that matches the channel question: each bipartite cell contributes a vowel row and a consonant row for the same text, so they form a matched pair. The exact binomial on the discordant pairs (McNemar) asks **which mark gives way when only one does**, with a 95% interval bootstrapped over lemmas rather than cells, because the events are concentrated in a few verbs.
  - **Outputs**: `reports/consonant_analysis_report.md`, `reports/consonant_summary.csv`, `reports/consonant_lemma_breakdown.csv`, and `reports/consonant_paired_discordance.csv`.

---

### 3. Sampler Convergence & Health Diagnostics

* **`mcmc_convergence.R`**:
  - **Purpose**: Audits Stan MCMC health across every fitted model found in `fits/` to guarantee reliable posterior exploration.
  - **Metrics**: Max $\hat{R}$, percentage of parameters with $\hat{R} \le 1.01$, minimum Bulk-ESS, minimum Tail-ESS, divergent transitions, and maximum treedepth hits.
  - **Outputs**: `reports/mcmc_convergence_table.md` and `reports/mcmc_convergence.csv`.

---

### 4. Pre-Modeling Data & Target State Audits (Double Robustness)

* **`attrition_diagnostics.py`**:
  - **Purpose**: Audits the longitudinal data pipeline from raw corpus texts (~1050–1650 CE) to the final GAMM dataset.
  - **Outputs**: `reports/attrition_report.md` and `reports/attrition_summary.csv`.

* **`target_sensitivity.py`**:
  - **Purpose**: Double robustness check verifying that defining each verb's teleological target state via `max(date)` per lemma is not biased by varying document survival dates.
  - **Outputs**: `reports/target_sensitivity_report.md` and `reports/target_sensitivity_summary.csv`.

---

## 🚀 Execution & Replication Workflow

To reproduce the analysis and audit reports:

```bash
# 1. Run Consonant Channel Analysis (audit + paired within-cell channel test)
python analysis/consonant_analysis.py

# 2. Run Pre-Modeling Diagnostics & Double Robustness Checks
python analysis/attrition_diagnostics.py
python analysis/target_sensitivity.py

# 3. Fit Bayesian GAMM Models (Option A: Vowel-Only)
# Dry-run validation:
Rscript analysis/run_brms.R --dry-run

# Fast test run:
Rscript analysis/run_brms.R --test

# Full MCMC production run - this is the exact command the committed fits were made with.
# Note that --max_treedepth defaults to 10 in the script; the fits in fits/ used 12.
Rscript analysis/run_brms.R --chains 4 --iter 4000 --cores 4 --threads 4 --seed 97 --adapt_delta 0.99 --max_treedepth 12

# 4. Generate MCMC Convergence Summary Table
Rscript analysis/mcmc_convergence.R

# 5. Render LOO-CV Comparison, Hypothesis Tests & Publication Figures
Rscript -e "rmarkdown::render('analysis/analyze_models.Rmd')"
```
