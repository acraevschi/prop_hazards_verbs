# Consonant Channel Analysis

## Executive Summary

This report provides a dedicated empirical audit of the consonant channel (`consonant_bipartite`) in Middle High German (MHG) and Early New High German (ENHG) strong verbs. While the primary Bayesian GAMM models focus on the vocalic channel (`vowel_unipartite` vs `vowel_bipartite`), the consonant channel behaves differently and is reported separately here.

### Key Findings:
1. **Elevated Raw Leveling Rate**: The consonant channel exhibits an overall leveling rate of **7.35%** (65 / 884 observations), which is substantially higher than bipartite vowel leveling (**0.80%**, OR = 9.82) and unipartite vowel leveling (**2.04%**, OR = 3.8).
2. **All of it is Verner, by construction**: every paradigm in this channel was admitted by `step_2_establish_baseline` only after a shape test that separates grammatischer Wechsel from Auslautverhärtung. Verner leaves the past plural as the odd cell (*wesen* s ~ s ~ r, *quëden* t ~ t ~ d); devoicing leaves the past singular as the odd cell (*scheiden* d ~ t ~ d). The Class I verbs *snîden*, *lîden* and *mîden* are d ~ t ~ **t** - the plural shares the t - so their t ~ d is grammatischer Wechsel, not a spelling effect. Verner-admitted: **7.35%** (65 / 884). Devoicing-shaped: **0 observations**, as expected - a non-zero count here would mean the upstream rule had changed.
3. **High Concentration**: The largest contributor is *ziehen* (lemma 17, 52.3% of consonant events).
4. **Within the cell, the consonant gives way first**: on the 711 cells where both channels are informative, exactly one mark gives way in 53 of them, and it is the consonant in **90.6%** of those (lemma-clustered 95% CI 60.0%-100.0%). This is the comparison the channel question actually asks, and it is reported in section 5.

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
| 17 | *ziehen* | `g ~ h / χ ~ g, g ~ χ` | Verner (past sg ~ past pl) | 543 | 34 | 6.26% | 52.3% |
| 95 | *lîden* | `t ~ d` | Verner (medial, pres ~ past) | 115 | 10 | 8.70% | 15.4% |
| 145 | *lîhen* | `w ~ h` | Verner (medial, pres ~ past) | 11 | 9 | 81.82% | 13.8% |
| 216 | *verlieren* | `r ~ s / r ~ s, s ~ r` | Verner (past sg ~ past pl) | 52 | 6 | 11.54% | 9.2% |
| 119 | *genesen* | `r ~ s / r ~ s, s ~ r` | Verner (past sg ~ past pl) | 12 | 4 | 33.33% | 6.2% |
| 8 | *snîden* | `t ~ d` | Verner (medial, pres ~ past) | 45 | 1 | 2.22% | 1.5% |
| 219 | *zîhen* | `g ~ h / χ ~ g, h ~ g, g ~ h, g ~ χ` | Verner (past sg ~ past pl) | 10 | 1 | 10.00% | 1.5% |
| 330 | *kièsen* | `r ~ s / r ~ s, s ~ r` | Verner (past sg ~ past pl) | 75 | 0 | 0.00% | 0.0% |
| 149 | *mîden* | `t ~ d` | Verner (medial, pres ~ past) | 21 | 0 | 0.00% | 0.0% |

## 4. Statistical Contrast Analysis (Unpaired - Descriptive Only)

> **Read these as descriptive rates, not as tests.** The consonant rows and the vowel-bipartite rows are not independent samples: 711 of them are the *same cells*, each contributing one row to each channel. Fisher's exact test assumes independence, so the p-values below are far smaller than the evidence warrants, and the events are concentrated in a handful of verbs besides. The consonant-vs-vowel comparison is tested properly in section 5, which uses the pairing instead of ignoring it. The `Vowel Unipartite` row is a between-lemma contrast and is reported for scale only; the modelled version of that contrast is the GAMM in `analysis/run_brms.R`.

| Comparison | Group 1 Rate | Group 2 Rate | Odds Ratio | 95% Confidence Interval | p-value (Fisher) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Consonant Bipartite (All) vs. Vowel Unipartite | 7.35% | 2.04% | 3.80 | [2.89, 5.01] | 8.60e-17 |
| Consonant Bipartite (All) vs. Vowel Bipartite | 7.35% | 0.80% | 9.82 | [5.38, 17.92] | 9.45e-19 |
| Consonant Verner-admitted vs. Vowel Bipartite | 7.35% | 0.80% | 9.82 | [5.38, 17.92] | 9.45e-19 |
| Consonant Devoicing-shaped vs. Vowel Bipartite | - | - | - | - | no observations in one of the two groups |

## 5. Within-Cell Channel Asymmetry (Paired Design)

Sections 1-4 compare channels as if they were separate samples. They are not. Every bipartite cell carries a vowel row and a consonant row describing the same stretch of text, so the two are a matched pair: same verb, same document, same date, same scribe, same inflectional slot, same frequency. Everything the GAMM spends its covariates controlling for cancels by construction here. The question this design answers is not *how much* each channel levels, but **which mark gives way when only one of them does**.

Matched cells: **711** across **9** bipartite verbs.

| | Consonant resisted | Consonant leveled |
| :--- | :---: | :---: |
| **Vowel resisted** | 655 | 48 |
| **Vowel leveled** | 5 | 3 |

Concordant cells (both marks resisted, or both gave way) carry no information about direction, so the test is the exact binomial on the **53 discordant** cells - McNemar's test in its exact form.

| Quantity | Value |
| :--- | :--- |
| Discordant cells | 53 |
| Consonant gave way | 48 |
| Vowel gave way | 5 |
| P(the mark that gives way is the consonant) | **90.6%** |
| Exact binomial p (vs 50%) | 7.08e-10 |
| Lemma-clustered 95% CI | (60.0%, 100.0%) |
| Bootstrap draws reversing the direction | 1.1% |

The interval resamples **verbs**, not cells. The events are concentrated, and an interval built by resampling cells would count one verb's many documents as many independent facts. The clustered interval is therefore much wider than the exact p-value suggests, and it is the one to quote.

### 5.1 Where the discordant cells come from

| Lemma ID | Lemma | Paired Cells | Consonant Only | Vowel Only | Discordant | Share (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 17 | *ziehen* | 491 | 33 | 3 | 36 | 67.9% |
| 145 | *lîhen* | 9 | 6 | 0 | 6 | 11.3% |
| 216 | *verlieren* | 52 | 6 | 0 | 6 | 11.3% |
| 119 | *genesen* | 11 | 3 | 0 | 3 | 5.7% |
| 95 | *lîden* | 59 | 0 | 2 | 2 | 3.8% |
| 8 | *snîden* | 22 | 0 | 0 | 0 | 0.0% |
| 149 | *mîden* | 10 | 0 | 0 | 0 | 0.0% |
| 219 | *zîhen* | 3 | 0 | 0 | 0 | 0.0% |
| 330 | *kièsen* | 54 | 0 | 0 | 0 | 0.0% |

### 5.2 What this does and does not support

1. **It refines Paul rather than contradicting him.** Paul's argument is that two marks reinforce each other. If one of them erodes several times faster than the other, the bipartite state is transient and asymmetric: the grammatischer Wechsel is the weak link, and bipartite marking is a way-station rather than a stable configuration.
2. **It is a separate result from the GAMM, with a separate design.** The bipartite-vs-unipartite contrast is between lemmas and rests on few verbs. This one is within the cell. Neither is a robustness check on the other, and they should be reported as two findings, not one.
3. **It is concentrated.** Read section 5.1 before quoting the percentage. The clustered interval already reflects that concentration; the point estimate does not.
4. **It does not license adding the consonant channel to `marking_type` as a third level.** Unipartite verbs have no consonant rows by construction, so that level would have no comparison group; the paired rows would enter the GAMM as if independent; and one random-effect structure cannot serve a between-lemma and a within-cell contrast at once. That is why `run_brms.R` fits the vowel channel only.
