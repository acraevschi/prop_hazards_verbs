# Testing Paul’s Principle: Diachronic Paradigm Leveling in High German Strong Verbs

This repository contains the data extraction, morphological parsing, lemma linking, and Bayesian statistical modeling code for investigating **Hermann Paul’s Principle** in the historical development of High German strong verbs from Middle High German (MHG, ~1050–1350) to Early New High German (ENHG, ~1350–1650).

---

## 📌 Project Overview

Hermann Paul (1886) hypothesized that paradigms characterized by multiplexed/redundant irregularity (i.e. **bipartite marking** combining vocalic *Ablaut* and consonantal *Grammatischer Wechsel* originating from Verner’s Law) establish a higher cognitive threshold of resistance against analogical leveling compared to unipartite paradigms.

This project implements:
1. **Automated Cross-Corpus Lemma Linking**: Unifying lemmas across ReM (MHG) and ReF (ENHG) using DWDS etymological scraping, candidate ranking, and a Disjoint Set Union (DSU) graph algorithm.
2. **Lemma-Guided Root Extraction**: Parsing normalized historical orthography to isolate root vocalic nuclei and consonantal codas without erroneously stripping roots or structural consonants. Prefix segmentation is read off the corpus's own lemma strings, where ReM marks a prefix with a hyphen (`ge-winnen`) and leaves a bare stem unmarked (`gëben`); ENHG rows inherit the analysis through `lemma_id`. Phonotactic guards remain as a fallback for the lemmas ReM does not cover.
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
│   │   ├── lemma_id.csv                     <- Final discrete lemma family identifiers
│   │   ├── unimorph_deu.tsv                 <- Cached UniMorph deu release (checksum-verified)
│   │   ├── unimorph_mapping.json            <- lemma -> 3sg preterite, with source variants
│   │   ├── nhg_targets.csv                  <- Modern target forms consumed by step_3_establish_targets
│   │   └── nhg_targets_misses.csv           <- Lemmas the source could not cover, and what was tried
│   ├── normalize_data.py                    <- Step 3: Dialect/date mapping & token frequency thresholding
│   ├── dialect_mapping.json / date_mapping.json
│   ├── extract_nhg_preterites.py            <- Step 3b: Modern preterites from UniMorph (deu), pinned commit
│   ├── build_nhg_targets.py                 <- Step 3c: Assemble modern infinitive/preterite targets per lemma_id
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
│   ├── marking_type_summary.py              <- Marking-type counts & bipartite concentration, without Stan
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

### Stage 3b: Modern German Target Forms

The endpoint of leveling is the root the verb reached in modern German. The
corpus cannot supply it: 46 of the 88 modelled lemmas have no past-tense token
dated 1500 or later. The targets therefore come from a dictionary source.

```bash
# Fetch UniMorph deu at a pinned commit, verify its checksum, resolve variants
python data/extract_nhg_preterites.py

# Assemble one modern infinitive and preterite per lemma_id
python data/build_nhg_targets.py
```

*`nhg_preterite` comes from a single source, UniMorph `deu` (CC BY-SA 3.0), at a
pinned commit whose SHA-256 is verified on every run; the licence, citation and
variant policy are documented in the header of `data/extract_nhg_preterites.py`.
Where UniMorph lists more than one preterite, the choice is enumerated per lemma
in `VARIANT_POLICY` with a reason, and the rejected variants are carried into
`nhg_targets.csv` so `analysis/marking_type_summary.py --sensitivity` can re-run
the coding under them. Lemmas the source cannot cover are left empty and listed
in `data/lemmas/nhg_targets_misses.csv`; the corpus rule remains the fallback
for those.*

### Stage 4: Root Extraction, Baselines & Leveling Coding
```bash
# Extract roots, calculate pre-1200 anchors, and code leveling outcomes
python data/corpus_approach_coding.py
```

Three decisions in this stage are worth knowing before reading any number that comes out of it.

**Vowel length is normalized away** (`â` → `a`, `î` → `i`). ReM is edited with Lachmann circumflexes and ReF is not, so treating length as contrastive makes every Class IV and V past plural in ReF (`nâmen`, `gâben`, `wâren`) look like a leveling event the moment the corpus changes at 1350. Measured on this data, that reads as a jump from 16.1% before 1350 to 63.2% after it. Normalizing removes it. A residual discontinuity remains (3.5% to 14.3%) and cannot be separated from time, because ReM and ReF do not overlap in date.

**The past subparadigm therefore has no vocalic contrast in Classes IV and V**, whose ablaut was purely quantitative. Those verbs are recovered through the consonant channel instead: `wësen` is `NA` for the vowel (`a` against `a` is uninformative) and carries its signal in `was ~ wâren`.

**Bipartiteness is decided per pair, with a shape test on each clause.** The past indicative singular is the one endingless cell, so its coda is word-final and every consonant comparison involving it is ambiguous between Verner's Law and Auslautverhärtung. The two have opposite shapes: Verner makes the past *plural* the odd cell out (`wesen ~ was ~ wâren`, s ~ s ~ r), final devoicing makes the past *singular* the odd cell out (`scheiden ~ schiet ~ schieden`, d ~ t ~ d). `step_2_establish_baseline` tests for this directly. There is deliberately no "any Grammatischer Wechsel anywhere and any Ablaut anywhere" clause: nearly every strong verb has ablaut somewhere, so that condition reduces to "any consonant difference at all" and hands the treatment variable to extraction noise.

### Stage 5: Consonant Analysis & Data Diagnostics

1. **Consonant Channel Analysis** (`analysis/consonant_analysis.py`):
   - **Purpose**: Dedicated empirical audit and statistical comparison of the consonant channel (`consonant_bipartite`).
   - **Key Distinctions**: Separates true morphological leveling (*Grammatischer Wechsel* / Verner's Law in *ziehen*, *verlieren*, *genesen*, *zîhen*) from phonological / orthographic variation (*Auslautverhärtung* in *scheiden*, *lîden*, *snîden*, *mîden*, *lîhen*).
   - **Command**:
     ```bash
     python analysis/consonant_analysis.py
     ```
   - **Outputs**: `analysis/reports/consonant_analysis_report.md`, `analysis/reports/consonant_summary.csv`, and `analysis/reports/consonant_lemma_breakdown.csv`.

2. **Anchor & Target Attrition Diagnostics** (`analysis/attrition_diagnostics.py`):
   - **Purpose**: Audits the longitudinal data pipeline from raw texts (~1050–1650) to the final GAMM dataset.
   - **Command**:
     ```bash
     python analysis/attrition_diagnostics.py
     ```
   - **Outputs**: `analysis/reports/attrition_report.md` and `analysis/reports/attrition_summary.csv`.

3. **Marking-Type Summary** (`analysis/marking_type_summary.py`):
   - **Purpose**: Reports the `marking_type` counts, the bipartite leveling rate by period, and how the bipartite events are distributed over lemmas, without fitting anything.
   - **Command**:
     ```bash
     python analysis/marking_type_summary.py
     python analysis/marking_type_summary.py --sensitivity   # re-code with every VARIANT_POLICY choice flipped
     ```

4. **Target State Sensitivity Analysis (Double Robustness Check)** (`analysis/target_sensitivity.py`):
   - **Purpose**: Verifies that the operationalization of each verb's teleological target state (its morphological endpoint by the end of ENHG) is not biased by varying document survival dates.
   - **Command**:
     ```bash
     python analysis/target_sensitivity.py
     ```
   - **Outputs**: `analysis/reports/target_sensitivity_report.md` and `analysis/reports/target_sensitivity_summary.csv`.

### Stage 6: Bayesian Statistical Modeling & Evaluation (Option A: Vowel-Only GAMMs)

Once the data is verified and coded:

1. **Fit Bayesian GAMM Models** (`analysis/run_brms.R`):
   - Fits 5 Bayesian Generalized Additive Mixed Models using `brms` and Stan on the **vowel-only** dataset with `vowel_unipartite` as the reference baseline ($\beta_0$).
   - Serializes and saves the fitted model objects (`.rds`) directly into the `fits/` folder, with LOO-CV attached.
   - **CLI Options**:
     ```bash
     # Dry-run validation (checks stancode & data without sampling):
     Rscript analysis/run_brms.R --dry-run

     # Fast test run (2 chains, small iterations):
     Rscript analysis/run_brms.R --test

     # Full production run (4 chains, 8 total CPU threads):
     Rscript analysis/run_brms.R --chains 4 --iter 4000 --cores 4 --threads 2 --seed 97
     ```

2. **MCMC Convergence Diagnostics** (`analysis/mcmc_convergence.R`):
   - Audits Stan sampler health across all 5 fitted models in `fits/` to verify reliable posterior exploration.
   - Evaluates: $\max(\hat{R})$, percentage of parameters with $\hat{R} \le 1.01$, minimum Bulk-ESS, minimum Tail-ESS, divergent transitions, and maximum treedepth hits.
   - **Command**:
     ```bash
     Rscript analysis/mcmc_convergence.R
     ```
   - **Outputs**: `analysis/reports/mcmc_convergence_table.md` and `analysis/reports/mcmc_convergence.csv`.

3. **LOO-CV Model Comparison, Hypothesis Tests & Publication Figures** (`analysis/analyze_models.Rmd`):
   - Computes Leave-One-Out Cross-Validation (PSIS-LOO), evaluates Paul's Principle via evidence ratios and dynamic contrasts, extracts marginal effects, and exports publication figures to `figures/`.
   - **Command**:
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

