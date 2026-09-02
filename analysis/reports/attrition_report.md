# Anchor & Target Attrition Diagnostics Report

## Quick Overview in Simple Words

This report answers a simple but fundamental question: **How and why does our dataset shrink as we move from raw medieval manuscripts to the final statistical models?**

When studying historical language change across 600 years (Middle High German ~1050 to Early New High German ~1650), we cannot include every word blindly. A verb must have an attested starting form in early texts (pre-1200) and an attested ending form in late texts. We also have to filter out regular pronunciation changes (sound changes) so they are not mistaken for grammatical leveling.

```
[Raw Corpora] 2,945 MHG / 1,754 ENHG Surface Lemmas (299,631 Tokens)
      │
      ▼  (Step 1: DSU Clustering of prefixes & spelling variants)
[Unified Lemma Families] 455 Families
      │
      ▼  (Step 2: Frequency thresholding: drop lemmas with <= 10 tokens)
[Normalized Strong Lemmas] 291 Families (150,496 Tokens)
      │
      ▼  (Step 3: Pre-1200 MHG Start-State Baseline requirement)
[Anchored Lemma Families] 197 Families (48,273 Past Tokens)
      │
      ▼  (Step 4: Sound-Change Filtering & Alternation Eligibility)
[Final GAMM Dataset] 88 Families (13,184 Analyzed Observations)
```

---

## Conceptual & Methodological Details

1. **Longitudinal Lemma & Token Attrition**:
   - **DSU Graph Clustering (Step 1)**: In historical texts, prefixed verbs (*ansehen*, *vorsehen*, *übersehen*) and spelling variants (*sehn*, *sehen*) are widespread. We group all related variants into simplex lemma families using a Disjoint Set Union (DSU) graph algorithm, turning thousands of surface strings into 455 discrete verb families.
   - **Frequency Filtering (Step 2)**: Verbs with 10 or fewer total occurrences across the centuries are removed to eliminate scribal hapax legomena and statistical noise.
   - **Start-State Anchoring (Step 3)**: To know if a verb "leveled" (simplified its irregular alternations), we must know its starting form before the change happened. We use Middle High German texts written prior to 1200 CE to establish this baseline anchor for each dialect region.
   - **Modeling Eligibility (Step 4)**: Out of 197 anchored verbs, 88 possessed an active historical vowel or consonant alternation (*Ablaut* or *Grammatischer Wechsel*) that was eligible for leveling and not already resolved by regular sound change.

2. **Disentangling Regular Sound Change from Analogical Leveling**:
   - Regular phonological shifts (such as diphthongization MHG *î* > ENHG *ei* in Central German, or monophthongization MHG *uo* > ENHG *u* in Upper German) happen mechanically across all words in a dialect.
   - If an observed vowel changed merely because of a dialect sound change, it would be a false positive to code it as morphological leveling.
   - Using our sound-change dictionary (`vowel_changes.csv`), we successfully filtered **4,823 transitions** that were pure sound changes, isolating **532 genuine analogical leveling events** and **12,652 leveling resistance observations**.

---

## 1. Lemma Attrition Diagnostics

| Stage / Dataset | Unique Count | Notes / Filtering Criteria |
| :--- | :---: | :--- |
| **Raw MHG Corpus (ReM)** | 2,945 | Unique surface lemma strings extracted from ReM JSON |
| **Raw ENHG Corpus (ReF)** | 1,754 | Unique surface lemma strings extracted from ReF XML |
| **Unified Lemma Families (DSU)** | 456 | Connected components created via DWDS scraping + DSU |
| **Normalized Lemma Families** | 292 | Retained strong verb lemmas with token count > 10 |
| **Lemmas Attested Pre-1200** | 222 | Lemmas attested in MHG before 1200 CE |
| **Lemmas Successfully Anchored** | 197 | Modal vowels/codas resolved for pre-1200 baseline |
| **Lemmas in Final GAMM Model** | 106 | Lemmas contributing valid binary outcome observations |

> **Retention Note**: Out of 292 normalized lemma families, 197 (67.5%) were successfully anchored with pre-1200 baselines.

---

## 2. Token Attrition & Filtering Diagnostics

| Pipeline Stage | Token Count | Retention (%) | Notes |
| :--- | :---: | :---: | :--- |
| **Total Combined Tokens** | 299,631 | 100.0% | All extracted verbal tokens across both corpora |
| **Normalized Strong Tokens** | 150,496 | 50.2% | Strong verbs with mapped dialects, dates & principal parts |
| **Past Indicative Subset (Coded)** | 49,616 | 33.0% | Past Singular and Past Plural subparadigms |
| **Tokens with Valid Baseline** | 48,273 | 97.3% | Successfully matched to pre-1200 dialect anchor |
| **Tokens Dropped (Missing Baseline)** | 1,343 | 2.7% | Excluded due to no pre-1200 attestation in dialect |
| **Final Analyzed Rows in GAMM** | 16,931 | 35.1% | Reshaped, non-redundant observations for brms modeling |

---

## 3. Leveling vs. Sound Change Diagnostics

| Classification | Count | Description |
| :--- | :---: | :--- |
| **Transitions Filtered by Sound Change (Past)** | 7,425 | Past targets matching anchor via regular dialect sound change |
| **Transitions Filtered by Sound Change (Pres)** | 1,555 | Present targets matching anchor via regular dialect sound change |
| **Total Transitions Filtered as Sound Change** | 8,980 | Prevented from false positive leveling coding |
| **Genuine Analogical Leveling (`has_levelled = 1`)** | 332 | Token matched target and broke from historical anchor |
| **Preserved / Resisted Leveling (`has_levelled = 0`)** | 16,599 | Token maintained historical anchor state |

### Leveling Rate by Marking Type

| Marking Type | Total Observations | Leveled (1) | Resisted (0) | Leveling Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| `vowel_bipartite` | 1,503 | 13 | 1,490 | 0.86% |
| `vowel_unipartite` | 15,428 | 319 | 15,109 | 2.07% |

### Leveling Rate by Macro-Variety

| Variety | Total Observations | Leveled (1) | Resisted (0) | Leveling Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| Central German | 6,063 | 122 | 5,941 | 2.01% |
| Upper German | 10,868 | 210 | 10,658 | 1.93% |

---
*Report generated automatically by `analysis/attrition_diagnostics.py`.*
