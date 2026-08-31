# Testing Paul’s Principle: Diachronic Paradigm Leveling in High German Strong Verbs

This repository contains the data extraction, morphological parsing, lemma linking, and Bayesian statistical modeling code for investigating **Hermann Paul’s Principle** in the historical development of High German strong verbs from Middle High German (MHG, ~1050–1350) to Early New High German (ENHG, ~1350–1650).

---

## 📌 Project Overview

Hermann Paul (1886) hypothesized that paradigms characterized by multiplexed/redundant irregularity (i.e. **bipartite marking** combining vocalic *Ablaut* and consonantal *Grammatischer Wechsel* originating from Verner’s Law) establish a higher cognitive threshold of resistance against analogical leveling compared to unipartite paradigms.

This project implements:
1. **Automated Cross-Corpus Lemma Linking**: Unifying lemmas across ReM (MHG) and ReF (ENHG) using DWDS etymological scraping, candidate ranking, and a Disjoint Set Union (DSU) graph algorithm.
2. **Phonotactically Guarded Root Extraction**: Parsing normalized historical orthography to isolate root vocalic nuclei and consonantal codas without erroneously stripping roots or structural consonants.
3. **Diachronic Baseline & Leveling Coding**: Setting pre-1200 MHG baselines and ENHG teleological targets per dialect variety while filtering regular dialect sound changes.
4. **Bayesian Generalized Additive Mixed Modeling (GAMM)**: Estimating non-linear diachronic trajectories and interactive tensor products of time and frequency using `brms` and Stan.

---

## 📂 Repository Structure

```
prop_hazards_verbs/
├── README.md                                <- Project overview and pipeline documentation
├── environment.yml                          <- Conda environment definition (R + Python dependencies)
├── LICENSE                                  <- License file
│
├── data/                                    <- Data processing pipeline & corpus files
│   ├── extract_mhg_data.py                  <- Step 1a: Extract verbal tokens from ReM (MHG JSON)
│   ├── extract_enhg_data.py                 <- Step 1b: Extract verbal tokens from ReF (ENHG XML)
│   ├── clean_lemmas.py                      <- Step 2a: Query DWDS etymological entries for ENHG lemmas
│   ├── lemmas/                              <- Step 2b: Cross-corpus lemma reconciliation
│   │   ├── link_lemmas.py                   <- Scrape etymology, rank candidates, query LLM
│   │   ├── enhg_mhg_mapping.py              <- DSU graph algorithm to assign unified lemma_id
│   │   ├── enhg_mapping.json / mhg_mapping.json
│   │   ├── etymology_matches_manual.csv     <- Curated etymological cross-links
│   │   └── lemma_id.csv                     <- Final discrete lemma family identifiers
│   ├── normalize_data.py                    <- Step 3: Dialect/date mapping & token frequency thresholding
│   ├── dialect_mapping.json / date_mapping.json
│   ├── corpus_approach_coding.py            <- Step 4: Root extraction, pre-1200 baseline & leveling coding
│   ├── vowel_changes.csv                    <- Dialect-specific sound change dictionary
│   ├── combined_corpus.csv / combined_normalized_corpus.csv
│   └── coded_output.csv                     <- Coded dataset for statistical modeling
│
├── analysis/                                <- Statistical modeling & evaluation
│   ├── data_for_analysis.csv                <- Reshaped dataset with predictors and marking types
│   ├── run_brms.R                           <- Step 5: Fits Bayesian GAMMs via brms / Stan
│   ├── analyze_models.Rmd                   <- Step 6: LOO-CV comparison, hypothesis testing, plots
│   ├── mcmc_convergence.R                  <- MCMC convergence diagnostics table across 5 fits
│   ├── attrition_diagnostics.py             <- Lemma/token attrition & sound-change filter diagnostics
│   ├── target_sensitivity.py                <- Double robustness check for late ENHG target definitions
│   ├── check_alternation_patterns.ipynb     <- Diagnostic inspection of alternation patterns
│   └── reports/                             <- Diagnostic audit & convergence reports (.md, .csv)
│
├── fits/                                    <- Serialized Bayesian model objects (.rds)
├── figures/                                 <- Publication-ready plots & figures (.pdf, .png)
└── tests/                                   <- Unit tests for morphological extraction & guards
```

---

## 🚀 End-to-End Pipeline Execution

The data processing pipeline combines automated corpus extraction and phonotactic parsing with a curated manual verification step for cross-corpus etymological linking.

### Stage 1: Raw Corpus Extraction
```bash
# Extract verbal tokens, inflection tags, and metadata from raw corpora
python data/extract_mhg_data.py
python data/extract_enhg_data.py
```

### Stage 2: Etymological Lemma Linking & Manual Verification

Cross-corpus lemma linking resolves Middle High German (MHG) and Early New High German (ENHG) verb variants into shared lemma families:

1. **Automated Candidate Extraction & LLM Ranking**:
   ```bash
   python data/clean_lemmas.py
   python data/lemmas/link_lemmas.py
   ```
   *Scrapes DWDS etymological entries, computes Levenshtein distances against MHG lemmas, and queries an LLM to generate candidate antecedents, saved to `data/lemmas/etymology_matches.csv`.*

2. **Manual Verification Step**:
   *The automated LLM suggestions in `data/lemmas/etymology_matches.csv` are manually reviewed and cleaned by the researchers to eliminate false cognates and verify historical accuracy, producing `data/lemmas/etymology_matches_manual.csv`.*
   > **Note on Replication**: The verified file `data/lemmas/etymology_matches_manual.csv` is already included in this repository. If you are replicating the analysis, you do not need to re-run the manual curation and can proceed directly to the next step.

3. **Graph Unification (Disjoint Set Union)**:
   ```bash
   python data/lemmas/enhg_mhg_mapping.py
   ```
   *Applies a DSU graph algorithm to group simplex bases, derived prefixes, and cross-corpus links into discrete `lemma_id` identifiers, outputting `data/lemmas/lemma_id.csv` and `data/combined_corpus.csv`.*

### Stage 3: Normalization & Thresholding
```bash
# Map macro-varieties, dates, inflection categories, and drop rare lemmas (<= 10 tokens)
python data/normalize_data.py
```

### Stage 4: Root Extraction, Baselines & Leveling Coding
```bash
# Extract roots with phonotactic guards, calculate pre-1200 anchors, and code leveling outcomes
python data/corpus_approach_coding.py
```

### Stage 5: Data & Target Diagnostics (Pre-Modeling Audits)

Before fitting the statistical models, run the diagnostic tools to audit data retention, phonological sound-change filtering, and target state stability:

1. **Anchor & Target Attrition Diagnostics** (`analysis/attrition_diagnostics.py`):
   - **Purpose**: Audits the longitudinal data pipeline from raw texts (~1050–1650) to the final GAMM dataset.
   - **What it tracks**:
     - *Lemma Attrition*: Traces unique lemmas from raw ReM (MHG) and ReF (ENHG) through DSU unification, frequency filtering ($> 10$), pre-1200 MHG start-state anchoring, and final GAMM modeling.
     - *Token Attrition*: Quantifies tokens dropped due to unanchored baselines versus analyzable past-tense tokens.
     - *Sound Change vs. Leveling*: Audits the phonological filter (`vowel_changes.csv`), reporting how many transitions represent regular dialectal sound changes (e.g. UG *uo* > *u*, CG *î* > *ei*) rather than genuine analogical leveling.
   - **Command**:
     ```bash
     python analysis/attrition_diagnostics.py
     ```
   - **Outputs**: `analysis/reports/attrition_report.md` and `analysis/reports/attrition_summary.csv`.

2. **Target State Sensitivity Analysis (Double Robustness Check)** (`analysis/target_sensitivity.py`):
   - **Purpose**: Verifies that the operationalization of each verb's teleological target state (its morphological endpoint by the end of ENHG) is not biased by varying document survival dates.
   - **What it evaluates**:
     - Compares the baseline target (`max(date)` per lemma) against two late-text regimes:
       - *Variant 1 (Strict Late Subset)*: Only includes lemmas attested in texts dated $\ge 1500$.
       - *Variant 2 (Hybrid Fallback)*: Prioritizes texts dated $\ge 1500$ when available, falling back to $\max(\text{date})$ for lemmas whose records cease earlier.
     - Re-codes leveling outcomes across all past-tense tokens to measure concordance and label stability.
   - **Command**:
     ```bash
     python analysis/target_sensitivity.py
     ```
   - **Outputs**: `analysis/reports/target_sensitivity_report.md` and `analysis/reports/target_sensitivity_summary.csv`.

### Stage 6: Bayesian Statistical Modeling & Evaluation

Once the data is verified and coded:

1. **Fit Bayesian GAMM Models** (`analysis/run_brms.R`):
   - Fits the Bayesian Generalized Additive Mixed Models using `brms` and Stan (accounting for smooth temporal trajectories, interactive tensor products, random effects, and alternation controls).
   - Serializes and saves the fitted model objects (`.rds`) directly into the `fits/` folder.
   - **Configurable Options** (defaults: `--chains 4 --iter 4000 --cores 4 --threads 4 --adapt_delta 0.99`):
   ```bash
   # Run with default settings:
   Rscript analysis/run_brms.R

   # Or customize sampler / hardware parameters:
   Rscript analysis/run_brms.R --chains 4 --iter 4000 --cores 8 --threads 2
   ```

2. **MCMC Convergence Diagnostics** (`analysis/mcmc_convergence.R`):
   - Audits Stan sampler health across all 5 fitted models in `fits/` to verify reliable posterior exploration.
   - Evaluates: $\max(\hat{R})$, percentage of parameters with $\hat{R} \le 1.01$, minimum Bulk-ESS, minimum Tail-ESS, divergent transitions, and maximum treedepth hits.
   ```bash
   Rscript analysis/mcmc_convergence.R
   ```
   - **Outputs**: `analysis/reports/mcmc_convergence_table.md` and `analysis/reports/mcmc_convergence.csv`.

3. **LOO-CV Model Comparison, Hypothesis Tests & Publication Figures** (`analysis/analyze_models.Rmd`):
   - Computes exact Leave-One-Out Cross-Validation (LOO-CV), evaluates Paul's Principle via evidence ratios and dynamic contrasts, extracts marginal effects, and exports publication figures to `figures/`.
   ```bash
   Rscript -e "rmarkdown::render('analysis/analyze_models.Rmd')"
   ```

---

## 🧪 Running Unit Tests

To run the unit tests verifying root extraction phonotactic guards, Ablaut classes (I-VII), Grammatischer Wechsel pairs, and equivalence functions:
```bash
python -m unittest discover -s tests -v
```

---

## ⚙️ Environment Setup

Create and activate the conda environment:
```bash
conda env create --file environment.yml
conda activate prop_hazards_verbs
```

> **Note on Stan**: On macOS, ensure the Xcode command line tools are installed (`xcode-select --install`). On Linux/Debian, ensure build essentials are available (`sudo apt-get install build-essential`).

