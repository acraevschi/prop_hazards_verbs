# Consonant Channel Analysis

## Executive Summary

This report provides a dedicated empirical audit of the consonant channel (`consonant_bipartite`) in Middle High German (MHG) and Early New High German (ENHG) strong verbs. While the primary Bayesian GAMM models focus on the vocalic channel (`vowel_unipartite` vs `vowel_bipartite`), the consonant channel behaves differently and is reported separately here.

### Key Findings:
1. **Elevated Raw Leveling Rate**: The consonant channel exhibits an overall leveling rate of **5.61%** (68 / 1,212 observations), which is substantially higher than bipartite vowel leveling (**0.59%**, OR = 10.0) and unipartite vowel leveling (**0.99%**, OR = 5.94).
2. **All of it is Verner, by construction**: every paradigm in this channel was admitted by `step_2_establish_baseline` only after a shape test that separates grammatischer Wechsel from Auslautverhärtung. Verner leaves the past plural as the odd cell (*wesen* s ~ s ~ r, *quëden* t ~ t ~ d); devoicing leaves the past singular as the odd cell (*scheiden* d ~ t ~ d). The Class I verbs *snîden*, *lîden* and *mîden* are d ~ t ~ **t** - the plural shares the t - so their t ~ d is grammatischer Wechsel, not a spelling effect. Verner-admitted: **5.61%** (68 / 1,212). Devoicing-shaped: **0 observations**, as expected - a non-zero count here would mean the upstream rule had changed.
3. **High Concentration**: The largest contributor is *ziehen* (lemma 17, 50.0% of consonant events).
4. **Within the token, the consonant gives way first**: on the 1,083 tokens where both channels are informative, exactly one mark gives way in 61 of them, and it is the consonant in **83.6%** of those (lemma-clustered 95% CI 36.4%-100.0%). This is the comparison the channel question actually asks, and it is reported in section 5.

## 1. Overall Marking Type Leveling Rates

| Marking Type | Observations | Leveling Events | Leveling Rate (%) |
| :--- | :---: | :---: | :---: |
| `vowel_unipartite` | 35,413 | 351 | 0.99% |
| `vowel_bipartite` | 2,878 | 17 | 0.59% |
| `consonant_bipartite` | 1,212 | 68 | 5.61% |

## 2. Breakdown by the Admitting Clause of the Bipartite Rule

| Alternation Category | Lemmas | Observations | Leveled | Leveling Rate (%) | Share of Consonant Events |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Verner (medial, pres ~ past)** | 4 | 350 | 22 | 6.29% | 32.4% |
| **Verner (past sg ~ past pl)** | 5 | 862 | 46 | 5.34% | 67.6% |

## 3. Per-Lemma Consonant Breakdown

| Lemma ID | Lemma | Alternation Pattern | Category | Obs | Leveled | Rate (%) | Share (%) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| 17 | *ziehen* | `g ~ h / g ~ χ, χ ~ g` | Verner (past sg ~ past pl) | 636 | 34 | 5.35% | 50.0% |
| 144 | *lîhen* | `w ~ h` | Verner (medial, pres ~ past) | 13 | 11 | 84.62% | 16.2% |
| 94 | *lîden* | `t ~ d` | Verner (medial, pres ~ past) | 246 | 10 | 4.07% | 14.7% |
| 215 | *verlieren* | `r ~ s / s ~ r, r ~ s` | Verner (past sg ~ past pl) | 94 | 6 | 6.38% | 8.8% |
| 118 | *genesen* | `r ~ s / s ~ r, r ~ s` | Verner (past sg ~ past pl) | 20 | 5 | 25.00% | 7.4% |
| 8 | *snîden* | `t ~ d` | Verner (medial, pres ~ past) | 65 | 1 | 1.54% | 1.5% |
| 218 | *zîhen* | `g ~ h / h ~ g, g ~ χ, g ~ h, χ ~ g` | Verner (past sg ~ past pl) | 13 | 1 | 7.69% | 1.5% |
| 329 | *kièsen* | `r ~ s / s ~ r, r ~ s` | Verner (past sg ~ past pl) | 99 | 0 | 0.00% | 0.0% |
| 148 | *mîden* | `t ~ d` | Verner (medial, pres ~ past) | 26 | 0 | 0.00% | 0.0% |

## 4. Statistical Contrast Analysis (Unpaired - Descriptive Only)

> **Read these as descriptive rates, not as tests.** The consonant rows and the vowel-bipartite rows are not independent samples: 1,083 of them are the *same tokens*, each contributing one row to each channel. Fisher's exact test assumes independence, so the p-values below are far smaller than the evidence warrants, and the events are concentrated in a handful of verbs besides. The consonant-vs-vowel comparison is tested properly in section 5, which uses the pairing instead of ignoring it. The `Vowel Unipartite` row is a between-lemma contrast and is reported for scale only; the modelled version of that contrast is the GAMM in `analysis/run_brms.R`.

| Comparison | Group 1 Rate | Group 2 Rate | Odds Ratio | 95% Confidence Interval | p-value (Fisher) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Consonant Bipartite (All) vs. Vowel Unipartite | 5.61% | 0.99% | 5.94 | [4.55, 7.75] | 1.45e-27 |
| Consonant Bipartite (All) vs. Vowel Bipartite | 5.61% | 0.59% | 10.00 | [5.85, 17.1] | 3.67e-22 |
| Consonant Verner-admitted vs. Vowel Bipartite | 5.61% | 0.59% | 10.00 | [5.85, 17.1] | 3.67e-22 |
| Consonant Devoicing-shaped vs. Vowel Bipartite | - | - | - | - | no observations in one of the two groups |

## 5. Within-Token Channel Asymmetry (Paired Design)

Sections 1-4 compare channels as if they were separate samples. They are not. Every bipartite token carries a vowel row and a consonant row describing the same attestation, so the two are a matched pair: same observation_id, verb, document, date, scribe, inflectional slot, and frequency. Everything the GAMM spends its covariates controlling for cancels by construction here. The question this design answers is not *how much* each channel levels, but **which mark gives way when only one of them does**.

Matched tokens: **1,083** across **9** bipartite verbs.

| | Consonant resisted | Consonant leveled |
| :--- | :---: | :---: |
| **Vowel resisted** | 1,018 | 51 |
| **Vowel leveled** | 10 | 4 |

Concordant tokens (both marks resisted, or both gave way) carry no information about direction, so the test is the exact binomial on the **61 discordant** tokens - McNemar's test in its exact form.

| Quantity | Value |
| :--- | :--- |
| Discordant tokens | 61 |
| Consonant gave way | 51 |
| Vowel gave way | 10 |
| P(the mark that gives way is the consonant) | **83.6%** |
| Exact binomial p (vs 50%) | 9.62e-08 |
| Lemma-clustered 95% CI | (36.4%, 100.0%) |
| Bootstrap draws reversing the direction | 5.7% |

The interval resamples **verbs**, not tokens. The events are concentrated, and an interval built by resampling tokens would count one verb's many attestations as many independent facts. The clustered interval is therefore much wider than the exact p-value suggests, and it is the one to quote.

### 5.1 Where the discordant tokens come from

| Lemma ID | Lemma | Paired Tokens | Consonant Only | Vowel Only | Discordant | Share (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 17 | *ziehen* | 626 | 33 | 3 | 36 | 59.0% |
| 144 | *lîhen* | 12 | 8 | 0 | 8 | 13.1% |
| 94 | *lîden* | 169 | 0 | 7 | 7 | 11.5% |
| 215 | *verlieren* | 94 | 6 | 0 | 6 | 9.8% |
| 118 | *genesen* | 19 | 4 | 0 | 4 | 6.6% |
| 8 | *snîden* | 44 | 0 | 0 | 0 | 0.0% |
| 148 | *mîden* | 15 | 0 | 0 | 0 | 0.0% |
| 218 | *zîhen* | 5 | 0 | 0 | 0 | 0.0% |
| 329 | *kièsen* | 99 | 0 | 0 | 0 | 0.0% |

### 5.2 What this does and does not support

1. **It refines Paul rather than contradicting him.** Paul's argument is that two marks reinforce each other. If one of them erodes several times faster than the other, the bipartite state is transient and asymmetric: the grammatischer Wechsel is the weak link, and bipartite marking is a way-station rather than a stable configuration.
2. **It is a separate result from the GAMM, with a separate design.** The bipartite-vs-unipartite contrast is between lemmas and rests on few verbs. This one is within the token. Neither is a robustness check on the other, and they should be reported as two findings, not one.
3. **It is concentrated.** Read section 5.1 before quoting the percentage. The clustered interval already reflects that concentration; the point estimate does not.
4. **It does not license adding the consonant channel to `marking_type` as a third level.** Unipartite verbs have no consonant rows by construction, so that level would have no comparison group; the paired rows would enter the GAMM as if independent; and one random-effect structure cannot serve a between-lemma and a within-token contrast at once. That is why `run_brms.R` fits the vowel channel only.
