# Statistical Analysis & Diagnostics

This directory contains the statistical modeling, MCMC convergence diagnostics, data attrition audits, and sensitivity analyses for testing **Hermann Paul’s Principle** in High German strong verbs.

---

## 📂 Directory Layout

```
analysis/
├── README.md                                <- This documentation file
├── data_for_analysis.csv                    <- Explicit document-lemma-slot-outcome observations
│
├── 🧠 Core Bayesian Modeling Pipeline
│   ├── run_brms.R                           <- Fits 6 Bayesian GAMM models via brms / Stan (Option A: Vowel-Only)
│   ├── analyze_models.Rmd                   <- LOO-CV model comparison, hypothesis testing, figure exports
│   └── analyze_models.html                  <- Rendered R Markdown analysis report
│
├── 🔬 Consonant Channel & Mechanism Audits
│   ├── consonant_analysis.py                <- Consonant channel audit + within-token paired channel test
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
    ├── regeneration_report.md               <- Task 2 provenance, old/new comparison, and source map
    ├── consonant_analysis_report.md         <- Comprehensive report on consonant channel leveling
    ├── consonant_summary.csv                <- Mechanism summary (Morphological GW vs Orthographic)
    ├── consonant_lemma_breakdown.csv        <- Per-lemma consonant leveling counts and rates
    ├── attrition_report.md                  <- Full funnel report on lemma/token attrition
    ├── attrition_summary.csv                <- Key attrition metrics table
    ├── target_sensitivity_report.md         <- Report on target state stability & label concordance
    ├── target_sensitivity_summary.csv       <- Outcome codability and concordance denominators
    ├── target_sensitivity_targets.csv       <- Direct target identity with missing groups retained
    ├── marking_type_report.md               <- Marking, concentration, lîhen, and variant audit
    ├── stage_counts.csv                     <- Stage-by-stage token, lemma, and document counts
    ├── baseline_anchor_support.csv          <- Support for every production anchor cell
    ├── model_predictor_distribution.csv     <- Model sample and predictor support
    ├── model_comparison.csv                 <- PSIS-LOO ordering from the rendered analysis
    ├── psis_loo_diagnostics.csv             <- Pareto-k bins for every fit
    ├── posterior_fixed_effects.csv          <- Fixed-effect posterior summaries
    ├── evidence_ratios.csv                  <- Directional hypothesis evidence ratio
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
  - **Observation unit**: repeated identical outcomes are removed explicitly by `document_id × lemma_id × std_infl × has_levelled`. A mixed cell contributes one Bernoulli row for each outcome state, independent of how often either state occurs. `leveled_tokens` and `preserved_tokens` are audit columns only; `document_id` supplies the document random effect. The separate consonant analysis remains paired by token-level `observation_id`.
  - **CLI Flags**: Supports `--prepare-only` (rebuilds and validates model data without Stan), `--test` (fast test fit), and `--dry-run` (validates formulas and Stan code without sampling) alongside `--chains`, `--iter`, `--cores`, `--threads`, `--backend`, and `--overwrite`.
  - **Outputs**: Serialized `.rds` model objects in `fits/`, a provenance sidecar for each fresh fit, and prepared modeling data in `analysis/data_for_analysis.csv`.

#### Reporting the within-document collapse

The response is not a token-level proportion and is not a binomial count. Source
tokens are first assigned to a corpus-qualified document, unified lemma family,
and inflectional slot. Repeated tokens with the same outcome are then collapsed
by taking
`distinct(document_id, lemma_id, std_infl, has_levelled)`. Consequently, a
document–lemma–slot cell containing only preserved tokens contributes one `0`, a
cell containing only levelled tokens contributes one `1`, and a genuinely mixed
cell contributes one `0` and one `1`. Ten repetitions of the same state therefore
have exactly the same likelihood contribution as one attestation of that state.
The original `leveled_tokens`, `preserved_tokens`, and `n_tokens` values are kept
in the exported table to audit the collapse, but are not supplied as trials,
weights, or responses to the models.

A concise methods description is: “We removed repeated identical outcomes within
each document–lemma–inflection cell. Cells containing only one outcome state
contributed one Bernoulli observation, whereas cells containing both preserved
and levelled forms contributed one observation of each; within-state token
multiplicity did not weight the likelihood.”

* **`analyze_models.Rmd`**:
  - **Purpose**: Comprehensive post-processing, Leave-One-Out Cross-Validation (PSIS-LOO), MCMC convergence tables, hypothesis testing (`marking_typevowel_bipartite < 0`), and publication figure generation.
  - **Outputs**: Publication-ready PDF figures in `figures/` (`fixed_effects.pdf`, `bi-uni_diff.pdf`, `leveling_trajectories.pdf`, etc.) and rendered HTML report `analyze_models.html`.

---

### 2. Dedicated Consonant Analysis

* **`consonant_analysis.py`**:
  - **Purpose**: Standalone module and CLI tool auditing the consonant channel on source-token observations.
  - **Category is derived from the paradigm, not hand-listed**: `classify_consonant_lemma()` reads the same anchors and the same `diff_cons_*` flags, through the same clauses, that `step_2_establish_baseline` used to admit the paradigm. The two therefore cannot drift apart. Because the upstream shape test already rejects *Auslautverhärtung* (*scheiden* d ~ t ~ d), **every paradigm here is grammatischer Wechsel by construction**; the devoicing category is empty and is printed as a tripwire on the rule, not as a finding about the language.
  - **Two designs, kept apart**:
    - *Sections 1–4* report unpaired rates and odds ratios. These are **descriptive only** — the consonant rows and the vowel-bipartite rows are largely measurements of the same tokens, so Fisher's exact test understates the uncertainty badly.
    - *Section 5* is the design that matches the channel question: each bipartite `observation_id` contributes one vowel row and one consonant row for the same attestation, so they form a matched pair. The exact binomial on the discordant pairs (McNemar) asks **which mark gives way when only one does**, with a 95% interval bootstrapped over lemmas rather than tokens, because the events are concentrated in a few verbs.
  - **Outputs**: `reports/consonant_analysis_report.md`, `reports/consonant_summary.csv`, `reports/consonant_lemma_breakdown.csv`, and `reports/consonant_paired_discordance.csv`.

---

### 3. Sampler Convergence & Health Diagnostics

* **`mcmc_convergence.R`**:
  - **Purpose**: Audits Stan MCMC health across every fitted model found in `fits/` to guarantee reliable posterior exploration.
  - **Metrics**: Max $\hat{R}$, percentage of parameters with $\hat{R} \le 1.01$, minimum Bulk-ESS, minimum Tail-ESS, divergent transitions, E-BFMI, and hits at the configured maximum treedepth.
  - **Outputs**: `reports/mcmc_convergence_table.md` and `reports/mcmc_convergence.csv`.

---

### 4. Pre-Modeling Data & Target State Audits (Double Robustness)

* **`attrition_diagnostics.py`**:
  - **Purpose**: Audits the longitudinal data pipeline with explicit mapping, frequency, missing-metadata, anchor-support, target-source, corpus/document, channel-outcome, and predictor denominators. Sound-change exclusions reuse production's protected contrasts.
  - **Outputs**: `reports/attrition_report.md`, `reports/attrition_summary.csv`, and the supporting audit CSVs listed above.

* **`target_sensitivity.py`**:
  - **Purpose**: Compares the production modern-first target against strict late-corpus and per-tense fallback definitions. Direct target identity, missing target groups, codability, and outcome agreement among shared codable observations remain distinct.
  - **Outputs**: `reports/target_sensitivity_report.md`, `reports/target_sensitivity_summary.csv`, and `reports/target_sensitivity_targets.csv`.

---

## 🚀 Execution & Replication Workflow

To reproduce the analysis and audit reports:

```bash
# 1. Run Consonant Channel Analysis (audit + paired within-token channel test)
python analysis/consonant_analysis.py

# 2. Run Pre-Modeling Diagnostics & Double Robustness Checks
python analysis/attrition_diagnostics.py
python analysis/target_sensitivity.py

# 3. Fit Bayesian GAMM Models (Option A: Vowel-Only)
# Prepare the explicitly deduplicated model table without Stan:
Rscript analysis/run_brms.R --prepare-only

# Dry-run validation:
Rscript analysis/run_brms.R --dry-run

# Fast test run:
Rscript analysis/run_brms.R --test

# Full MCMC production run. This resolves to 4 chains, 4,000 iterations,
# 2,000 warmup, seed 97, adapt_delta 0.99, and treedepth 10.
Rscript analysis/run_brms.R --chains 4 --cores 4 --threads 4

# 4. Generate MCMC Convergence Summary Table
Rscript analysis/mcmc_convergence.R

# 5. Render LOO-CV Comparison, Hypothesis Tests & Publication Figures
Rscript -e "rmarkdown::render('analysis/analyze_models.Rmd')"
```
