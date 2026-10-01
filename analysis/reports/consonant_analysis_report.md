# Consonant Channel Analysis

## Executive Summary

This report provides a dedicated empirical audit of the consonant channel (`consonant_bipartite`) in Middle High German (MHG) and Early New High German (ENHG) strong verbs. While the primary Bayesian GAMM models focus on the vocalic channel (`vowel_unipartite` vs `vowel_bipartite`), the consonant channel behaves differently and is reported separately here.

### Key Findings:
1. **Elevated Raw Leveling Rate**: The consonant channel exhibits an overall leveling rate of **10.57%** (155 / 1,466 observations), which is substantially higher than bipartite vowel leveling (**1.70%**, OR = 6.86) and unipartite vowel leveling (**1.37%**, OR = 8.51).
2. **All of it is Verner, by construction**: every paradigm in this channel was admitted by `step_2_establish_baseline` only after a shape test that separates grammatischer Wechsel from Auslautverhärtung. Verner leaves the past plural as the odd cell (*wesen* s ~ s ~ r, *quëden* t ~ t ~ d); devoicing leaves the past singular as the odd cell (*scheiden* d ~ t ~ d). The Class I verbs *snîden*, *lîden* and *mîden* are d ~ t ~ **t** - the plural shares the t - so their t ~ d is grammatischer Wechsel, not a spelling effect. Verner-admitted: **10.57%** (155 / 1,466). Devoicing-shaped: **0 observations**, as expected - a non-zero count here would mean the upstream rule had changed.
3. **High Concentration**: The largest contributor is *ziehen* (lemma 17, 65.8% of consonant events).
4. **Observed within-token asymmetry favors consonant leveling, with substantial uncertainty across verbs**: on the 1,321 tokens where both channels are informative, exactly one mark levels in 131 of them, and it is the consonant in **84.7%** of those (lemma-clustered 95% CI 42.9%-100.0%). Because that interval includes the 50% null, the data do not establish a verb-general directional asymmetry. This comparison is reported in section 5.

## 1. Overall Marking Type Leveling Rates

| Marking Type | Observations | Leveling Events | Leveling Rate (%) |
| :--- | :---: | :---: | :---: |
| `vowel_unipartite` | 42,028 | 576 | 1.37% |
| `vowel_bipartite` | 3,362 | 57 | 1.70% |
| `consonant_bipartite` | 1,466 | 155 | 10.57% |

## 2. Breakdown by the Admitting Clause of the Bipartite Rule

| Alternation Category | Lemmas | Observations | Leveled | Leveling Rate (%) | Share of Consonant Events |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Verner (medial, pres ~ past)** | 4 | 364 | 27 | 7.42% | 17.4% |
| **Verner (past sg ~ past pl)** | 5 | 1,102 | 128 | 11.62% | 82.6% |

## 3. Per-Lemma Consonant Breakdown

| Lemma ID | Lemma | Alternation Pattern | Category | Obs | Leveled | Rate (%) | Share (%) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| 17 | *ziehen* | `g ~ h / g ~ χ, χ ~ g` | Verner (past sg ~ past pl) | 857 | 102 | 11.90% | 65.8% |
| 215 | *verlieren* | `r ~ s / s ~ r, r ~ s` | Verner (past sg ~ past pl) | 109 | 20 | 18.35% | 12.9% |
| 144 | *lîhen* | `w ~ h` | Verner (medial, pres ~ past) | 15 | 13 | 86.67% | 8.4% |
| 94 | *lîden* | `t ~ d` | Verner (medial, pres ~ past) | 256 | 12 | 4.69% | 7.7% |
| 118 | *genesen* | `r ~ s / s ~ r, r ~ s` | Verner (past sg ~ past pl) | 20 | 5 | 25.00% | 3.2% |
| 8 | *snîden* | `t ~ d` | Verner (medial, pres ~ past) | 66 | 2 | 3.03% | 1.3% |
| 218 | *zîhen* | `g ~ h / g ~ h, g ~ χ, h ~ g, χ ~ g` | Verner (past sg ~ past pl) | 17 | 1 | 5.88% | 0.6% |
| 329 | *kièsen* | `r ~ s / s ~ r, r ~ s` | Verner (past sg ~ past pl) | 99 | 0 | 0.00% | 0.0% |
| 148 | *mîden* | `t ~ d` | Verner (medial, pres ~ past) | 27 | 0 | 0.00% | 0.0% |

## 4. Statistical Contrast Analysis (Unpaired - Descriptive Only)

> **Read these as descriptive rates, not as tests.** The consonant rows and the vowel-bipartite rows are not independent samples: 1,321 of them are the *same tokens*, each contributing one row to each channel. Fisher's exact test assumes independence, so the p-values below are far smaller than the evidence warrants, and the events are concentrated in a handful of verbs besides. The consonant-vs-vowel comparison is tested properly in section 5, which uses the pairing instead of ignoring it. The `Vowel Unipartite` row is a between-lemma contrast and is reported for scale only; the modelled version of that contrast is the GAMM in `analysis/run_brms.R`.

| Comparison | Group 1 Rate | Group 2 Rate | Odds Ratio | 95% Confidence Interval | p-value (Fisher) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Consonant Bipartite (All) vs. Vowel Unipartite | 10.57% | 1.37% | 8.51 | [7.07, 10.24] | 1.38e-77 |
| Consonant Bipartite (All) vs. Vowel Bipartite | 10.57% | 1.70% | 6.86 | [5.03, 9.35] | 2.70e-39 |
| Consonant Verner-admitted vs. Vowel Bipartite | 10.57% | 1.70% | 6.86 | [5.03, 9.35] | 2.70e-39 |
| Consonant Devoicing-shaped vs. Vowel Bipartite | - | - | - | - | no observations in one of the two groups |

## 5. Within-Token Channel Asymmetry (Paired Design)

Sections 1-4 compare channels as if they were separate samples. They are not. Every bipartite token carries a vowel row and a consonant row describing the same attestation, so the two are a matched pair: same observation_id, verb, document, date, scribe, inflectional slot, and frequency. Everything the GAMM spends its covariates controlling for cancels by construction here. The question this design answers is not *how much* each channel levels, but **which mark gives way when only one of them does**.

Matched tokens: **1,321** across **9** bipartite verbs.

| | Consonant resisted | Consonant leveled |
| :--- | :---: | :---: |
| **Vowel resisted** | 1,162 | 111 |
| **Vowel leveled** | 20 | 28 |

Concordant tokens (both marks resisted, or both leveled) carry no information about direction, so the test is the exact binomial on the **131 discordant** tokens - McNemar's test in its exact form. This exact p-value is conditional on treating the discordant tokens as independent. Pairing controls the comparison within a token, but it does not make repeated tokens from the same verb or document independent.

| Quantity | Value |
| :--- | :--- |
| Discordant tokens | 131 |
| Consonant gave way | 111 |
| Vowel gave way | 20 |
| P(the mark that gives way is the consonant) | **84.7%** |
| Exact binomial p (vs 50%) | 1.76e-16 |
| Lemma-clustered 95% CI | (42.9%, 100.0%) |
| Bootstrap draws reversing the direction | 3.5% |

The interval resamples **verbs**, not tokens. The events are concentrated, and an interval built by resampling tokens would count one verb's many attestations as many independent facts. The clustered interval is therefore much wider than the exact p-value suggests, includes the 50% null, and is the one to quote. It addresses concentration by verb; it does not turn the observed asymmetry into evidence about which channel changed earlier or at a faster historical rate.

### 5.1 Where the discordant tokens come from

| Lemma ID | Lemma | Paired Tokens | Consonant Only | Vowel Only | Discordant | Share (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 17 | *ziehen* | 840 | 78 | 12 | 90 | 68.7% |
| 215 | *verlieren* | 109 | 20 | 0 | 20 | 15.3% |
| 94 | *lîden* | 173 | 0 | 8 | 8 | 6.1% |
| 144 | *lîhen* | 14 | 8 | 0 | 8 | 6.1% |
| 118 | *genesen* | 19 | 4 | 0 | 4 | 3.1% |
| 8 | *snîden* | 45 | 1 | 0 | 1 | 0.8% |
| 148 | *mîden* | 16 | 0 | 0 | 0 | 0.0% |
| 218 | *zîhen* | 6 | 0 | 0 | 0 | 0.0% |
| 329 | *kièsen* | 99 | 0 | 0 | 0 | 0.0% |

### 5.2 What this does and does not support

1. **It is descriptive evidence of channel asymmetry, not temporal sequence or rate.** The point estimate favors consonant-only leveling among discordant tokens, but the lemma-clustered interval includes 50%. These data therefore do not establish that the consonant channel gives way first, erodes faster, or is a general weak link across verbs.
2. **It is a separate result from the GAMM, with a separate design.** The bipartite-vs-unipartite contrast is between lemmas and rests on few verbs. This one is within the token. Neither is a robustness check on the other, and they should be reported as two findings, not one.
3. **It is concentrated.** Read section 5.1 before quoting the percentage. The clustered interval already reflects that concentration; the point estimate does not.
4. **It does not license adding the consonant channel to `marking_type` as a third level.** Unipartite verbs have no consonant rows by construction, so that level would have no comparison group; the paired rows would enter the GAMM as if independent; and one random-effect structure cannot serve a between-lemma and a within-token contrast at once. That is why `run_brms.R` fits the vowel channel only.
