# Consonant Channel Analysis: Morphological Leveling vs. Orthographic Variation

## Executive Summary

This report provides a dedicated empirical audit of the consonant channel (`consonant_bipartite`) in Middle High German (MHG) and Early New High German (ENHG) strong verbs. While the primary Bayesian GAMM models focus on the vocalic channel (`vowel_unipartite` vs `vowel_bipartite`), the consonant channel exhibits distinct historical, phonological, and orthographic dynamics.

### Key Findings:
1. **Elevated Raw Leveling Rate**: The consonant channel exhibits an overall leveling rate of **8.02%** (65 / 810 observations), which is substantially higher than bipartite vowel leveling (**0.86%**, OR = 10.0) and unipartite vowel leveling (**2.07%**, OR = 4.13).
2. **Dual Mechanism Breakdown**: Consonant alternations conflate two fundamentally different historical processes:
   - **True Morphological Leveling (*Grammatischer Wechsel* / Verner's Law)**: Genuine stem alternations (*s ~ r* in *verlieren*, *genesen*; *g ~ h/χ* in *ziehen*, *zîhen*). Leveling rate: **7.29%**.
   - **Orthographic / Phonological Variation (*Auslautverhärtung*)**: Coda alternations (*t ~ d* in *scheiden*, *lîden*, *snîden*; *w ~ h* in *lîhen*), driven by final devoicing and scribal standardizations. Leveling rate: **10.36%**.
3. **High Concentration**: The largest morphological contributor is *ziehen* (lemma 17, 52.3%), while the largest orthographic contributor is *lîden* (lemma 95, 15.4%).

## 1. Overall Marking Type Leveling Rates

| Marking Type | Observations | Leveling Events | Leveling Rate (%) |
| :--- | :---: | :---: | :---: |
| `vowel_unipartite` | 15,428 | 319 | 2.07% |
| `vowel_bipartite` | 1,503 | 13 | 0.86% |
| `consonant_bipartite` | 810 | 65 | 8.02% |

## 2. Morphological vs. Orthographic Mechanism Breakdown

| Alternation Category | Lemmas | Observations | Leveled | Leveling Rate (%) | Share of Consonant Events |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Morphological GW** | 4 | 617 | 45 | 7.29% | 69.2% |
| **Orthographic / Devoicing** | 5 | 193 | 20 | 10.36% | 30.8% |

## 3. Per-Lemma Consonant Breakdown

| Lemma ID | Lemma | Alternation Pattern | Category | Obs | Leveled | Rate (%) | Share (%) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| 17 | *ziehen* | `g ~ h / χ ~ g, g ~ χ` | Morphological GW | 543 | 34 | 6.26% | 52.3% |
| 95 | *lîden* | `t ~ d` | Orthographic / Devoicing | 115 | 10 | 8.70% | 15.4% |
| 145 | *lîhen* | `w ~ h` | Orthographic / Devoicing | 11 | 9 | 81.82% | 13.8% |
| 216 | *verlieren* | `r ~ s / r ~ s, s ~ r` | Morphological GW | 52 | 6 | 11.54% | 9.2% |
| 119 | *genesen* | `r ~ s / r ~ s, s ~ r` | Morphological GW | 12 | 4 | 33.33% | 6.2% |
| 8 | *snîden* | `t ~ d` | Orthographic / Devoicing | 45 | 1 | 2.22% | 1.5% |
| 219 | *zîhen* | `g ~ h / χ ~ g, g ~ χ, h ~ g, g ~ h` | Morphological GW | 10 | 1 | 10.00% | 1.5% |
| 149 | *mîden* | `t ~ d` | Orthographic / Devoicing | 21 | 0 | 0.00% | 0.0% |
| 193 | *sièden* | `t ~ d` | Orthographic / Devoicing | 1 | 0 | 0.00% | 0.0% |

## 4. Statistical Contrast Analysis

| Comparison | Group 1 Rate | Group 2 Rate | Odds Ratio | 95% Confidence Interval | p-value (Fisher) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Consonant Bipartite (All) vs. Vowel Unipartite | 8.02% | 2.07% | 4.13 | [3.13, 5.45] | 2.30e-18 |
| Consonant Bipartite (All) vs. Vowel Bipartite | 8.02% | 0.86% | 10.00 | [5.48, 18.26] | 5.44e-19 |
| Consonant Morphological (Verner's Law) vs. Vowel Bipartite | 7.29% | 0.86% | 9.02 | [4.83, 16.84] | 1.23e-14 |
| Consonant Orthographic (Auslautverhärtung) vs. Vowel Bipartite | 10.36% | 0.86% | 13.25 | [6.48, 27.11] | 7.97e-12 |

## 5. Methodological & Theoretical Implications

1. **Justification for Option A (Vowel-Only Primary Model)**:
   - Combining consonant marking with vowel marking into a single treatment factor introduces substantial noise because consonant leveling is heavily confounded by orthographic spelling shifts (*Auslautverhärtung* in *scheiden* and *lîden*).
   - By separating the consonant channel into this dedicated analysis and estimating a pure vowel-only GAMM, the test of Paul's Principle tests morphological Ablaut resistance against an unambiguous unipartite control baseline.
2. **Grammatischer Wechsel Trajectory**:
   - True Grammatischer Wechsel verbs (*ziehen*, *verlieren*, *genesen*, *zîhen*) undergo progressive analogical leveling across the MHG-to-ENHG transition as the plural/participle voiced consonants generalize into the singular preterite.
   - This confirms that consonantal stem alternations operate under distinct phonetic and morphosyntactic pressures compared to vowel ablaut grades.
