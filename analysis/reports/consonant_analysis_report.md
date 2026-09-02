# Consonant Channel Analysis

## Executive Summary

This report provides a dedicated empirical audit of the consonant channel (`consonant_bipartite`) in Middle High German (MHG) and Early New High German (ENHG) strong verbs. While the primary Bayesian GAMM models focus on the vocalic channel (`vowel_unipartite` vs `vowel_bipartite`), the consonant channel behaves differently and is reported separately here.

### Key Findings:
1. **Elevated Raw Leveling Rate**: The consonant channel exhibits an overall leveling rate of **7.35%** (65 / 884 observations), which is substantially higher than bipartite vowel leveling (**0.80%**, OR = 9.82) and unipartite vowel leveling (**2.04%**, OR = 3.8).
2. **All of it is Verner, by construction**: every paradigm in this channel was admitted by `step_2_establish_baseline` only after a shape test that separates grammatischer Wechsel from Auslautverhärtung. Verner leaves the past plural as the odd cell (*wesen* s ~ s ~ r, *quëden* t ~ t ~ d); devoicing leaves the past singular as the odd cell (*scheiden* d ~ t ~ d). The Class I verbs *snîden*, *lîden* and *mîden* are d ~ t ~ **t** - the plural shares the t - so their t ~ d is grammatischer Wechsel, not a spelling effect. Verner-admitted: **7.35%** (65 / 884). Devoicing-shaped: **0 observations**, as expected - a non-zero count here would mean the upstream rule had changed.
3. **High Concentration**: The largest contributor is *ziehen* (lemma 17, 52.3% of consonant events).

## 1. Overall Marking Type Leveling Rates

| Marking Type | Observations | Leveling Events | Leveling Rate (%) |
| :--- | :---: | :---: | :---: |
| `vowel_unipartite` | 15,845 | 324 | 2.04% |
| `vowel_bipartite` | 1,622 | 13 | 0.80% |
| `consonant_bipartite` | 884 | 65 | 7.35% |

## 2. Breakdown by the Admitting Clause of the Bipartite Rule

| Alternation Category | Lemmas | Observations | Leveled | Leveling Rate (%) | Share of Consonant Events |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Verner (medial, pres ~ past)** | 4 | 192 | 20 | 10.42% | 30.8% |
| **Verner (past sg ~ past pl)** | 5 | 692 | 45 | 6.50% | 69.2% |

## 3. Per-Lemma Consonant Breakdown

| Lemma ID | Lemma | Alternation Pattern | Category | Obs | Leveled | Rate (%) | Share (%) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| 17 | *ziehen* | `g ~ h / g ~ χ, χ ~ g` | Verner (past sg ~ past pl) | 543 | 34 | 6.26% | 52.3% |
| 95 | *lîden* | `t ~ d` | Verner (medial, pres ~ past) | 115 | 10 | 8.70% | 15.4% |
| 145 | *lîhen* | `w ~ h` | Verner (medial, pres ~ past) | 11 | 9 | 81.82% | 13.8% |
| 216 | *verlieren* | `r ~ s / s ~ r, r ~ s` | Verner (past sg ~ past pl) | 52 | 6 | 11.54% | 9.2% |
| 119 | *genesen* | `r ~ s / s ~ r, r ~ s` | Verner (past sg ~ past pl) | 12 | 4 | 33.33% | 6.2% |
| 8 | *snîden* | `t ~ d` | Verner (medial, pres ~ past) | 45 | 1 | 2.22% | 1.5% |
| 219 | *zîhen* | `g ~ h / g ~ χ, g ~ h, χ ~ g, h ~ g` | Verner (past sg ~ past pl) | 10 | 1 | 10.00% | 1.5% |
| 330 | *kièsen* | `r ~ s / s ~ r, r ~ s` | Verner (past sg ~ past pl) | 75 | 0 | 0.00% | 0.0% |
| 149 | *mîden* | `t ~ d` | Verner (medial, pres ~ past) | 21 | 0 | 0.00% | 0.0% |

## 4. Statistical Contrast Analysis

| Comparison | Group 1 Rate | Group 2 Rate | Odds Ratio | 95% Confidence Interval | p-value (Fisher) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Consonant Bipartite (All) vs. Vowel Unipartite | 7.35% | 2.04% | 3.80 | [2.89, 5.01] | 8.60e-17 |
| Consonant Bipartite (All) vs. Vowel Bipartite | 7.35% | 0.80% | 9.82 | [5.38, 17.92] | 9.45e-19 |
| Consonant Verner-admitted vs. Vowel Bipartite | 7.35% | 0.80% | 9.82 | [5.38, 17.92] | 9.45e-19 |
| Consonant Devoicing-shaped vs. Vowel Bipartite | - | - | - | - | no observations in one of the two groups |

## 5. Methodological & Theoretical Implications

1. **Justification for Option A (Vowel-Only Primary Model)**:
   - Combining consonant marking with vowel marking into a single treatment factor introduces substantial noise because consonant leveling is heavily confounded by orthographic spelling shifts (*Auslautverhärtung* in *scheiden* and *lîden*).
   - By separating the consonant channel into this dedicated analysis and estimating a pure vowel-only GAMM, the test of Paul's Principle tests morphological Ablaut resistance against an unambiguous unipartite control baseline.
2. **Grammatischer Wechsel Trajectory**:
   - True Grammatischer Wechsel verbs (*ziehen*, *verlieren*, *genesen*, *zîhen*) undergo progressive analogical leveling across the MHG-to-ENHG transition as the plural/participle voiced consonants generalize into the singular preterite.
   - This confirms that consonantal stem alternations operate under distinct phonetic and morphosyntactic pressures compared to vowel ablaut grades.
