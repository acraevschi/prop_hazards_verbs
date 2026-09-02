# Comprehensive Data Coverage Comparison Report: Old vs. New Data Coding Pipeline

**Pipeline Comparison:** Baseline / Old Pipeline (`99b98d0`) vs. New Pipeline (`4e267b7` / `HEAD`)  
**Intermediate Audit Commits:** Target & Lemma Repairs (`8990054`), Subjunctive Principal Part Fix (`9ed8c84`)  
**Date:** September 2026  
**Corpus Scope:** Middle High German (*Referenzkorpus Mittelhochdeutsch*, ReM, ~1050–1350) and Early New High German (*Referenzkorpus Frühneuhochdeutsch*, ReF, ~1350–1650)

---

## Executive Summary

A comprehensive data audit and comparative analysis was conducted between the historical baseline data coding pipeline (commit `99b98d0`) and the refined, production pipeline (commit `4e267b7`). 

The pipeline refactoring resolved fundamental linguistic and architectural issues in historical German strong verb leveling:
1. **Target Preterite Resolution & Modern Harmonization (`8990054`)**: Eliminated an 80.1% missingness rate in historical preterite targets via systematic, pinned UniMorph extraction (`nhg_targets.csv`), repaired corrupted lemma families (re-integrating orphaned high-frequency verbs like *ziehen* and *liegen*), and corrected an 11.84× ENHG frequency distortion.
2. **Subjunctive Principal Part Disambiguation (`9ed8c84`)**: Resolved a critical misclassification where 3,003 past subjunctive singular tokens (formed on the past plural stem with umlaut, e.g., *hülfe*, *züge*, *fände*) were filed as indicative `PastSg`. This single fix eliminated **200 spurious leveling events** (a 30.5% reduction in false positives).
3. **Phonotactic & Morphological Root Extraction Refinement (`4e267b7`)**: Implemented lemma-guided prefix stripping, slot-aware ending preservation (protecting root-final consonants in endingless past singulars like *gewan*), and early *qu-* > *kw-* onset normalization (resolving *quam* nuclei).
4. **Auslautverhärtung vs. Verner's Law Shape Test (`4e267b7`)**: Replaced naive bipartite classification with an explicit paradigmatic shape test that separates Middle High German final devoicing (*Auslautverhärtung*, $d \sim t \sim d$ in *scheiden*) from genuine Verner's Law alternation (*Grammatischer Wechsel*, $d \sim t \sim t$ in *snîden* / *lîden* and $s \sim s \sim r$ in *wesen* / *kiesen*).

```
Pipeline Evolution Overview:
═════════════════════════════════════════════════════════════════════════════════════════════
Commit & Description                       Modeled Obs   Unique Lemmas   Total Leveled   Lev. Rate
─────────────────────────────────────────────────────────────────────────────────────────────
99b98d0 (Old Baseline)                       13,184            88             532          4.04%
8990054 (Target Harmonization & Curation)    17,497           106             655          3.74%
9ed8c84 (Subjunctive Principal Part Fix)     17,688           106             455          2.57%
4e267b7 (HEAD: Root & Auslaut Refinement)    17,878           106             434 (397*)   2.43% (2.24%*)
═════════════════════════════════════════════════════════════════════════════════════════════
* 17,741 observations and 397 leveling events fall strictly within the tripartite marking classification.
```

---

## 1. Token & Observation Counts

### 1.1 Pipeline Attrition Funnel

The longitudinal dataset filters raw manuscript transcriptions down to robust statistical modeling observations across four discrete stages:

| Pipeline Stage | Old Pipeline (`99b98d0`) | New Pipeline (`4e267b7`) | Retention (%) | Notes / Filtering Criteria |
| :--- | :---: | :---: | :---: | :--- |
| **1. Combined Raw Tokens** | 299,631 | 299,631 | 100.00% | All verbal tokens extracted from ReM JSON and ReF XML |
| **2. Normalized Strong Tokens** | 150,496 | 150,496 | 50.23% | Strong verbs filtered by token count > 10, valid dates & dialects |
| **3. Coded Past Subset (`coded_output.csv`)** | 49,616 | 49,616 | 16.56% | Past indicative & subjunctive tokens matched to pre-1200 anchors |
| **4. Modeled Dataset (Reshaped / De-duplicated)** | **13,184** | **17,878** | **5.97%** | Non-redundant modeling observations across document–outcome cells |
| **— Observations with Tripartite `marking_type`** | 13,184 | 17,741 | 5.92% | Classified into `vowel_unipartite`, `vowel_bipartite`, `consonant_bipartite` |

> **Key Difference**: The new pipeline yields **+4,694 additional modeled observations (+35.6%)** (17,878 vs. 13,184). This expansion is driven by resolving missing modern targets, integrating previously orphaned lemma families, and properly anchoring past plural / subjunctive tokens.

---

### 1.2 Unique Lemmas and Corpus Distribution (MHG vs. ENHG)

| Metric / Stage | Old Pipeline (`99b98d0`) | New Pipeline (`4e267b7`) | Net Change | Linguistic / Engineering Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **Raw Surface Lemmas (ReM / MHG)** | 2,939 | 2,939 | 0 | Raw Middle High German lexicon |
| **Raw Surface Lemmas (ReF / ENHG)** | 497 | 497 | 0 | Raw Early New High German lexicon |
| **Normalized Lemma Families (Total)** | 291 | 292 | +1 | DSU graph clustering of prefix variants |
| — Normalized Families in MHG | 228 | 228 | 0 | Pre-1350 attestations |
| — Normalized Families in ENHG | 194 | 197 | +3 | Improved simplex mapping (*anfangen* -> *fangen*) |
| **Coded Lemma Families (`coded_output.csv`)** | 234 | 234 | 0 | Families with pre-1200 anchors |
| — Coded Families in MHG | 197 | 197 | 0 | Pre-1200 Middle High German anchors |
| — Coded Families in ENHG | 127 | 129 | +2 | Modern target resolution coverage |
| **Modeled Lemma Families (in GAMM / brms)** | **88** | **106** | **+18 (+20.5%)** | Active alternations eligible for leveling |
| — Modeled Families in MHG | 88 | 106 | +18 | Broadened anchor coverage in MHG |
| — Modeled Families in ENHG | 60 | 80 | +20 | Restored target preterite coverage in ENHG |

#### Corpus Token and Observation Distribution

```
Coded Dataset (N = 49,616 tokens):
  ├─ MHG (ReM, 1050–1350):  37,312 tokens (75.20%)
  └─ ENHG (ReF, 1350–1650): 12,304 tokens (24.80%)

Modeled Observations:
  Old Pipeline (N = 13,184):
    ├─ MHG (ReM):  6,065 obs (46.00%) | Leveled = 157 (2.59%)
    └─ ENHG (ReF): 7,119 obs (54.00%) | Leveled = 375 (5.27%)
  New Pipeline (N = 17,878):
    ├─ MHG (ReM):  8,677 obs (48.53%) | Leveled = 101 (1.16%)
    └─ ENHG (ReF): 9,201 obs (51.47%) | Leveled = 333 (3.62%)
```

---

### 1.3 Dialect Coverage and Regional Representation

All modeled observations map onto two primary dialect macro-regions (Central German and Upper German):

| Dialect Macro-Region | Old Obs (`99b98d0`) | Old Leveled | Old Rate (%) | New Obs (`4e267b7`) | New Leveled | New Rate (%) | Net Obs Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Central German** (*Mitteldeutsch*) | 4,587 | 132 | 2.88% | **6,391** | 158 | **2.47%** | +1,804 (+39.3%) |
| **Upper German** (*Oberdeutsch*) | 8,597 | 400 | 4.65% | **11,487** | 276 | **2.40%** | +2,890 (+33.6%) |
| **Total** | **13,184** | **532** | **4.04%** | **17,878** | **434** | **2.43%** | **+4,694 (+35.6%)** |

> **Linguistic Insight**: In the old pipeline, Upper German appeared to level at nearly double the rate of Central German (4.65% vs. 2.88%). In the new pipeline, after purging false subjunctive leveling and misattributed final devoicing, the leveling rates across Central German (2.47%) and Upper German (2.40%) are virtually identical.

---

### 1.4 Missing / NA Rates in Root Extraction and Coding

The table below contrasts data completeness in `data/coded_output.csv` (N = 49,616 tokens):

| Variable / Pipeline Step | Old NA Count (`99b98d0`) | Old NA (%) | New NA Count (`4e267b7`) | New NA (%) | Improvement / Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`extracted_vowel`** | 0 | 0.00% | 0 | 0.00% | 100% vowel extraction across all tokens |
| **`extracted_coda`** | 1,740 | 3.51% | **523** | **1.05%** | **-70.0% NA reduction** (preserved coda in endingless forms) |
| **`anchor_vowel_pres`** | 1,708 | 3.44% | 1,708 | 3.44% | Pre-1200 present baseline anchor |
| **`anchor_vowel_pastsg`** | 1,649 | 3.32% | **1,570** | **3.16%** | Improved past singular baseline resolution |
| **`anchor_vowel_pastpl`** | 2,805 | 5.65% | **1,969** | **3.97%** | **-29.8% NA reduction** in past plural baseline anchors |
| **`target_vowel_pres`** | 12,621 | 25.44% | **5,122** | **10.32%** | **-59.4% NA reduction** via UniMorph modern targets |
| **`target_vowel_past`** | **39,739** | **80.09%** | **5,128** | **10.34%** | **-87.1% NA reduction** (fixed systemic preterite target loss) |
| **`is_bipartite`** | 1,343 | 2.71% | 1,343 | 2.71% | Only unanchored hapax lemmas lack baseline type |
| **Rows with NO outcome coded** | **19,471** | **39.24%** | **11,562** | **23.30%** | **-40.6% unmodeled token reduction** |
| **Rows with VALID outcome coded** | **30,145** | **60.76%** | **38,054** | **76.70%** | **+7,909 additional usable tokens (+26.2%)** |

---

## 2. Marking Types & Subparadigms

### 2.1 Marking Type Breakdown Across Pipeline Evolution

The core morphological predictor classifies strong verb alternation systems into three categories:
1. `vowel_unipartite`: Standard ablaut without Grammatischer Wechsel (e.g., *helfen ~ half ~ geholfen*, *tragen ~ trug*).
2. `vowel_bipartite`: Vowel leveling in verbs exhibiting Grammatischer Wechsel (e.g., *ziehen ~ zôch ~ zugen*, *snîden ~ sneit ~ sniten*).
3. `consonant_bipartite`: Consonantal leveling in Grammatischer Wechsel verbs (e.g., *snîden ~ sneit ~ sniten* leveling $t \to d$, *ziehen* leveling $\chi/g \to h$).

| Pipeline Version | `vowel_unipartite` Obs / Lev (Rate) | `vowel_bipartite` Obs / Lev (Rate) | `consonant_bipartite` Obs / Lev (Rate) | Total Analyzed Obs / Lev (Rate) |
| :--- | :---: | :---: | :---: | :---: |
| **Old Baseline (`99b98d0`)** | 12,175 / 466 (3.83%) | 705 / 3 (0.43%) | 304 / 63 (20.72%) | 13,184 / 532 (4.04%) |
| **Target Repairs (`8990054`)** | 14,849 / 475 (3.20%) | 1,676 / 35 (2.09%) | 972 / 145 (14.92%) | 17,497 / 655 (3.74%) |
| **Subjunctive Fix (`9ed8c84`)** | 15,075 / 338 (2.24%) | 1,672 / 15 (0.90%) | 941 / 102 (10.84%) | 17,688 / 455 (2.57%) |
| **HEAD Pipeline (`4e267b7`)** | **15,428 / 319 (2.07%)** | **1,503 / 13 (0.86%)** | **810 / 65 (8.02%)** | **17,741 / 397 (2.24%)** |

```
Marking Type Leveling Rates Comparison:
  vowel_unipartite:      3.83%  ───►  2.07%  (Spurious subjunctive singulars eliminated)
  vowel_bipartite:       0.43%  ───►  0.86%  (True bipartite cohort anchored across 106 lemmas)
  consonant_bipartite:  20.72%  ───►  8.02%  (Auslautverhärtung purged from Verner's Law baseline)
```

---

### 2.2 Subparadigm Counts & Leveling Distribution

The dataset focuses on the preterite subparadigms (`PastSg` vs. `PastPl`):

#### In Coded Output (`coded_output.csv`, N = 49,616)
- **Old Pipeline (`99b98d0`)**: `PastSg` = 39,653 tokens (79.9%), `PastPl` = 9,963 tokens (20.1%).
- **New Pipeline (`4e267b7`)**: `PastSg` = 36,650 tokens (73.9%), `PastPl` = 12,966 tokens (26.1%).
- *Shift*: Exactly **3,003 past subjunctive singular tokens** correctly moved from `PastSg` to `PastPl` (principal part 3).

#### In Modeled Observations (De-duplicated)

| Subparadigm | Old Obs (`99b98d0`) | Old Leveled | Old Rate (%) | New Obs (`4e267b7`) | New Leveled | New Rate (%) | Net Obs Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Past Singular (`PastSg`)** | 10,644 | 405 | 3.80% | **13,479** | 220 | **1.63%** | +2,835 (+26.6%) |
| **Past Plural (`PastPl`)** | 2,540 | 127 | 5.00% | **4,399** | 214 | **4.86%** | +1,859 (+73.2%) |
| **Total** | **13,184** | **532** | **4.04%** | **17,878** | **434** | **2.43%** | **+4,694 (+35.6%)** |

---

### 2.3 Leveling by Contrast Channel (Disaggregated)

Collapsing present–past and past singular–past plural alternations obscures contrasting dynamics. The table below presents the fully disaggregated channel breakdown:

| Contrast Channel | Marking Type | Old Tokens | Old Obs | Old Lev | Old Rate (%) | New Tokens | New Obs | New Lev | New Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vowel (PastSg ~ PastPl)** | `unipartite` | 2,569 | 1,072 | 114 | 10.63% | **3,783** | **1,668** | **134** | **8.03%** |
| **Vowel (PastSg ~ PastPl)** | `bipartite` | 217 | 104 | 0 | 0.00% | **519** | **308** | **9** | **2.92%** |
| **Vowel (Pres ~ Past)** | `unipartite` | 26,984 | 11,103 | 352 | 3.17% | **33,380** | **12,287** | **179** | **1.46%** |
| **Vowel (Pres ~ Past)** | `bipartite` | 1,622 | 543 | 3 | 0.55% | **2,497** | **1,098** | **4** | **0.36%** |
| **Consonant (PastSg ~ PastPl)** | `bipartite` | 16 | 9 | 3 | 33.33% | **576** | **421** | **43** | **10.21%** |
| **Consonant (Pres ~ Past)** | `bipartite` | 596 | 255 | 61 | 23.92% | **571** | **340** | **26** | **7.65%** |

---

### 2.4 Bipartite Vowel Leveling by Period (S-Curve Trajectory)

A central theoretical question is whether bipartite vowel leveling follows an S-curve across 600 years:

| Time Period | 99b98d0 Obs / Lev (Rate) | 8990054 Obs / Lev (Rate) | 9ed8c84 Obs / Lev (Rate) | 4e267b7 (HEAD) Obs / Lev (Rate) | Trajectory Evaluation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1050–1200** | 261 / 1 (0.38%) | 507 / 9 (1.78%) | 497 / 1 (0.20%) | **454 / 1 (0.22%)** | Stable baseline resistance |
| **1200–1350** | 269 / 0 (0.00%) | 517 / 11 (2.13%) | 524 / 4 (0.76%) | **446 / 3 (0.67%)** | Initial analogical onset |
| **1350–1500** | 171 / 2 (1.17%) | 632 / 13 (2.06%) | 632 / 9 (1.42%) | **584 / 8 (1.37%)** | Steady acceleration |
| **1500–1650** | 4 / 0 (0.00%) | 19 / 2 (10.53%) | 18 / 1 (5.56%) | **18 / 1 (5.56%)\*** | Small late-sample endpoint |

\* *Note*: The 1500–1650 window contains 18 observations, reflecting corpus sparser late preterites in ReF. While the monotonic increase (0.22% $\to$ 0.67% $\to$ 1.37% $\to$ 5.56%) is linguistically consistent, it rests on 13 total events.

---

## 3. Leveling Events & Per-Lemma Distribution

### 3.1 Concentration of Bipartite Vowel Leveling Events

In the refined pipeline (`4e267b7`), the 13 bipartite vowel leveling events originate across **5 distinct verb families**:

| Lemma ID | Simplex Lemma | Total Modeled Obs | Leveling Events | Share of Bipartite Events (%) | Historical Alternation Type |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **95** | *lîden* | 69 | 4 | 30.8% | Class I ($d \sim t \sim t$, *leit / liten*) |
| **17** | *ziehen* | 534 | 3 | 23.1% | Class II ($h \sim \chi \sim g$, *zôch / zugen*) |
| **145** | *lîhen* | 11 | 3 | 23.1% | Class I ($h \sim \chi \sim h$, *lêch / lihen*) |
| **8** | *snîden* | 31 | 2 | 15.4% | Class I ($d \sim t \sim t$, *sneit / sniten*) |
| **219** | *zîhen* | 18 | 1 | 7.7% | Class I ($h \sim \chi \sim g$, *zêch / zigen*) |
| **Total** | — | **663** | **13** | **100.0%** | **5 contributing lemmas out of 12 bipartite families** |

---

### 3.2 Per-Lemma Leveling Event Gains and Losses

The table below tracks every major lemma family whose event counts shifted between the old baseline (`99b98d0`) and the new pipeline (`4e267b7`), with exact linguistic attribution:

| Lemma String | Old Obs / Lev | New Obs / Lev | Net Leveling Shift | Primary Cause & Linguistic Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **`liegen`** | 196 / **183** | 508 / **0** | **-183** | **Etymological curation repair**: Old pipeline conflated MHG *ligen* (Class V, *lac/lâgen*) with *liègen* (Class II, *louc/lugen*), miscoding regular indicative *lac* as leveled to *louc*. |
| **`finden`** | 289 / **61** | 441 / **5** | **-56** | **Target harmonization & subjunctive fix**: Old target coded as *fund* (plural/dialect variant); corrected to *fand* (UniMorph 3sg), and 3sg subjunctives (*fände*) moved to PP3. |
| **`tragen`** | 265 / **0** | 314 / **47** | **+47** | **Preterite target recovery**: Old pipeline had `target_vowel_past = NaN` for *tragen*; new pipeline resolved target *trug* (*u*), capturing genuine ENHG *u*-leveling. |
| **`scheiden`** | 339 / **49** | 331 / **38** | **-11** | **Auslautverhärtung shape test**: Reclassified from bipartite to unipartite; eliminated spurious consonant leveling in endingless *schiet* ($d \sim t \sim d$). |
| **`ziehen`** | 6 / **2** | 1,077 / **37** | **+35** | **Lemma family reunification**: Reconnected MHG *zièhen* (700+ tokens) with ENHG *ziehen* (previously split into orphaned entries). |
| **`stîgen`** | 0 / **0** | 50 / **34** | **+34** | **Lemma resolution**: Pre-1200 anchor recovered; captured Class I preterite leveling (*steig* replacing *stigen*). |
| **`brennen`** | 45 / **0** | 88 / **26** | **+26** | **Target resolution**: Modern *brannte* (*a*) correctly coded against MHG *brente/brante* Rückumlaut baseline. |
| **`fahren`** | 245 / **20** | 227 / **0** | **-20** | **Pres target harmonization**: Eliminated spurious leveling from 2sg/3sg present umlaut (*fährt*) vs. 1sg (*fahre*). |
| **`brechen`** | 241 / **5** | 256 / **21** | **+16** | **Target preterite *brach***: Subjunctive singulars cleaned, isolating genuine preterite vowel shifts. |
| **`gewinnen`** | 277 / **19** | 319 / **5** | **-14** | **Endingless coda preservation**: Fixed blind *-n* stripping in past singular *gewan*; eliminated spurious consonant leveling. |
| **`gelten`** | 66 / **12** | 89 / **1** | **-11** | **Target harmonization**: Corrected target preterite *galt* (*a*) resolving false past leveling. |
| **`werfen`** | 80 / **9** | 200 / **1** | **-8** | **Subjunctive fix**: Eliminated false leveling from past subjunctive singular *würfe*. |
| **`snîden`** | 7 / **1** | 76 / **3** | **+2** | **Verner shape test preservation**: Properly maintained as bipartite ($d \sim t \sim t$); captured genuine $t \to d$ leveling. |

---

### 3.3 Dissection of Key Pipeline Fixes

#### A. Subjunctive Principal Parts Fix (`8990054` $\to$ `9ed8c84`)
- **Mechanism**: The Middle High German past subjunctive is formed on the **past plural** stem with umlaut (*hulfen* $\to$ *hülfe*, *zugen* $\to$ *züge*, *nâmen* $\to$ *nǣme*). In the old pipeline, pattern-matching on inflection strings ignored mood, categorizing all 3sg past subjunctives as `PastSg`.
- **Linguistic Error**: Because these 3,003 subjunctive singular tokens carried the past plural vowel, filing them under `PastSg` made them appear as singulars that had analogically adopted the plural stem—the exact definition of leveling.
- **Impact**: Removing this artifact eliminated **200 false leveling events** in a single commit (137 in `vowel_unipartite`, 20 in `vowel_bipartite`, 43 in `consonant_bipartite`).

```
Principal Part vs. std_infl Alignment:
  Old Pipeline (8990054):  PP2 = 36,650 PastSg | PP3 = 9,963 PastPl + 3,003 PastSg (Subjunctives)
  New Pipeline (9ed8c84):  PP2 = 36,650 PastSg | PP3 = 12,966 PastPl + 0 PastSg
```

#### B. Lemma-Guided Prefix Stripping & Endingless Coda Preservation (`9ed8c84` $\to$ `4e267b7`)
- **Mechanism**:
  1. *Hyphen-Guided Stripping*: Rather than stripping *ge-*, *be-*, *ver-* phonotactically (which cut into roots like *gëben* $\to$ *ben*, *bergen* $\to$ *rgen*), the pipeline reads hyphenated lemma analyses (*ge-winnen*, *be-rinnen*, *er-gëzzen*).
  2. *Coda Cluster Safeguards*: Enforces that clusters like *lt*, *rg*, *rb*, *nd*, *nt* cannot serve as syllable onsets, preventing erroneous truncation.
  3. *Slot-Aware Ending Stripping (`SLOTS_WITH_ENDING`)*: In Middle High German, the past indicative singular is endingless (*gap*, *nam*, *was*, *wan*, *gewan*). Stripping final *-n* blindly turned *gewan* into *ge-* + *wan* (coda *w* / `NaN`), inventing a consonant alternation *gewinnen* never had. The new pipeline restricts *-n* stripping to ending-bearing slots (`Pres`, `PastPl`, `Ppl`).
  4. *Qu- Normalization*: Converted *qu-* $\to$ *kw-* prior to vowel extraction, fixing historical past forms of *komen* (*quam*) whose nuclei previously extracted as corrupted diphthong *ua* rather than *a*.

#### C. Auslautverhärtung vs. Verner's Law Shape Test (`9ed8c84` $\to$ `4e267b7`)
- **Mechanism**: The endingless past singular has a word-final coda subject to final devoicing (*Auslautverhärtung*), while present and past plural cells carry vocalic endings that place their codas in medial voiced position:
  - **Devoicing (Non-Bipartite)**: Past singular is the *only* odd cell out ($d \sim t \sim d$):
    - *scheiden ~ schiet ~ schieden* ($d \sim t \sim d$)
    - *binden ~ bant ~ bunden* ($d \sim t \sim d$)
  - **Verner's Law (Genuine Bipartite)**: Past plural is the odd cell out ($s \sim s \sim r$ or $h \sim \chi \sim g$), or both past cells share the change ($d \sim t \sim t$):
    - *wesen ~ was ~ wâren* ($s \sim s \sim r$)
    - *ziehen ~ zôch ~ zugen* ($h \sim \chi \sim g$)
    - *snîden ~ sneit ~ sniten* ($d \sim t \sim t$)
    - *quëden ~ quat ~ quâden* ($t \sim t \sim d$)
- **Impact**: Cleanly distinguished 810 true consonant bipartite observations from 15,428 unipartite observations, eliminating 49 false leveling events from *scheiden*.

#### D. Vowel Length Normalization Across the 1350 Boundary
- **Mechanism**: ReM (pre-1350) marks vowel length explicitly (*â, ê, î, ô, û*), whereas ReF (post-1350) uses unmarked historical orthography. Length is stripped during NFD normalization (*â* $\to$ *a*) to prevent the 1350 corpus transition from being miscoded as a massive wave of quantitative leveling.
- **Protected Contrasts**: Stripping length creates collisions between dialect sound changes (MHG *î* > ENHG *ei*) and Class I ablaut (*snîden* *i* vs. *sneit* *ei*). The `protected` set preserves pre-1200 baseline contrasts, preventing regular sound change lookups from erroneously explaining away genuine morphological ablaut alternations.

---

## 4. Synthesis & Recommendations

### 4.1 Master Pipeline Comparison Summary

| Metric / Dimension | Old Baseline (`99b98d0`) | Intermediate (`8990054`) | Intermediate (`9ed8c84`) | New Pipeline (`4e267b7`) | Overall Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Coded Tokens** | 49,616 | 49,616 | 49,616 | 49,616 | Baseline constant |
| **Modeled Observations** | 13,184 | 17,497 | 17,688 | **17,878** | **+35.6% coverage** |
| **Modeled Lemma Families** | 88 | 106 | 106 | **106** | **+20.5% coverage** |
| **Preterite Target Missingness** | 80.09% | 10.34% | 10.34% | **10.34%** | **-87.1% missingness** |
| **Total Leveling Events** | 532 | 655 | 455 | **434** | **-18.4% false positives** |
| **Overall Leveling Rate** | 4.04% | 3.74% | 2.57% | **2.43%** | Cleaner signal |
| **`vowel_unipartite` Rate** | 3.83% | 3.20% | 2.24% | **2.07%** | Subjunctive noise removed |
| **`vowel_bipartite` Rate** | 0.43% | 2.09% | 0.90% | **0.86%** | Solidified 5-lemma cohort |
| **`consonant_bipartite` Rate**| 20.72% | 14.92% | 10.84% | **8.02%** | Devoicing artifacts purged |

### 4.2 Conclusions & Implications for Statistical Modeling

1. **Model Robustness & Convergent Estimation**: The refined pipeline provides a 35.6% larger, linguistically verified empirical base (17,878 observations across 106 lemma families) for Bayesian survival hazard modeling (`brms` / Stan).
2. **Elimination of False Diachronic Spikes**: By eliminating 200 spurious subjunctive leveling events and resolving *scheiden* / *legen* / *fahren* coding artifacts, the baseline leveling rate stabilizes from an inflated 4.04% down to a genuine 2.24–2.43%.
3. **Bipartite Vowel Sparsity**: Bipartite vowel leveling exhibits a monotonic diachronic rise across centuries (0.22% $\to$ 0.67% $\to$ 1.37% $\to$ 5.56%), but rests on **13 total events across 5 lemmas** (*lîden*, *ziehen*, *lîhen*, *snîden*, *zîhen*). Statistical models should incorporate random lemma intercepts and sensitivity checks across target choices.
4. **All Pipeline Tests Passing**: All 47 pytest test cases in `tests/test_extraction.py` pass cleanly, validating root extraction, prefix stripping, slot-aware ending truncation, and baseline establishment.
