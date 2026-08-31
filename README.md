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
│   └── check_alternation_patterns.ipynb     <- Diagnostic inspection of alternation patterns
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

### Stage 5: Bayesian Statistical Modeling & Analysis
```bash
# Fit Bayesian GAMM models using brms / Stan (saved to fits/)
Rscript analysis/run_brms.R

# Render model comparisons, hypothesis checks, and publication figures
Rscript -e "rmarkdown::render('analysis/analyze_models.Rmd')"
```

---

## 🧪 Running Unit Tests

To run the unit tests verifying root extraction phonotactic guards and equivalence functions:
```bash
python -m unittest discover -s tests
```

---

## ⚙️ Environment Setup

Create and activate the conda environment:
```bash
conda env create --file environment.yml
conda activate prop_hazards_verbs
```

> **Note on Stan**: On macOS, ensure the Xcode command line tools are installed (`xcode-select --install`). On Linux/Debian, ensure build essentials are available (`sudo apt-get install build-essential`).

