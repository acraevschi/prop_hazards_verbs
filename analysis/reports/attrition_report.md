# Attrition and Data-Support Audit

All counts below were regenerated from the pipeline artifacts. The baseline is
inclusive: MHG observations dated **1200 or earlier** (`date <= 1200`),
and it uses exactly `Pres, PastSg, PastPl`. Participles are not baseline
anchors because they enter neither production contrast.

## Stage counts

`denominator` names the immediately relevant universe for each row; the notes
state the gate. MHG and ENHG extraction rows are separate and are reconciled by
the combined-join row.

| stage | denominator | observations | source_tokens | observation_unit | surface_lemmas | lemma_families | documents | note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MHG extraction | 164,187 | 164,187 | 164,187 | token row | 2,945 | 0 | 405 | all extracted MHG strong-verb tokens |
| ENHG extraction | 166,800 | 166,800 | 166,800 | token row | 1,754 | 0 | 190 | all extracted ENHG strong-verb tokens |
| Combined join | 330,987 | 330,987 | 330,987 | token row | 4,671 | 455 | 595 | mapped and unmapped rows retained |
| Mapped join rows | 330,987 | 281,458 | 281,458 | token row | 3,406 | 455 | 594 | non-missing corpus-aware lemma_id |
| Frequency eligible | 281,458 | 280,870 | 280,870 | token row | 3,175 | 291 | 594 | family has more than 10 mapped tokens |
| Normalized | 280,870 | 142,045 | 142,045 | token row | 2,608 | 290 | 234 | date, variety, and principal part all resolved |
| Coding eligible | 142,045 | 108,389 | 108,389 | token row | 2,335 | 282 | 234 | excluded non-strong/irregular inflClass labels |
| Coded past tokens | 108,389 | 47,248 | 47,248 | token row | 1,587 | 233 | 226 | PastSg and PastPl rows emitted by outcome coder |
| Vowel model | 47,248 | 7,510 | 38,291 | document-lemma-slot-outcome presence | 124 | 124 | 226 | distinct document x lemma-family x slot x outcome rows |

## Join audit by corpus

| corpus | joined_rows | mapped_rows | unmapped_rows | denominator |
| :--- | :--- | :--- | :--- | :--- |
| ENHG | 166,800 | 117,281 | 49,519 | 166,800 |
| MHG | 164,187 | 164,177 | 10 | 164,187 |

## Corpus and document support

| stage | corpus | observations | source_tokens | observation_unit | documents | surface_lemmas | lemma_families |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| extracted | ENHG | 166,800 | 166,800 | token row | 190 | 1,754 | 0 |
| extracted | MHG | 164,187 | 164,187 | token row | 405 | 2,945 | 0 |
| combined | ENHG | 166,800 | 166,800 | token row | 190 | 1,754 | 227 |
| combined | MHG | 164,187 | 164,187 | token row | 405 | 2,945 | 367 |
| mapped | ENHG | 117,281 | 117,281 | token row | 189 | 490 | 227 |
| mapped | MHG | 164,177 | 164,177 | token row | 405 | 2,939 | 367 |
| normalized | ENHG | 48,830 | 48,830 | token row | 78 | 420 | 193 |
| normalized | MHG | 93,215 | 93,215 | token row | 156 | 2,207 | 227 |
| coding_eligible | ENHG | 31,539 | 31,539 | token row | 78 | 397 | 191 |
| coding_eligible | MHG | 76,850 | 76,850 | token row | 156 | 1,956 | 216 |
| coded | ENHG | 11,457 | 11,457 | token row | 75 | 262 | 126 |
| coded | MHG | 35,791 | 35,791 | token row | 151 | 1,337 | 197 |
| vowel_model | ENHG | 1,526 | 8,737 | document-lemma-slot-outcome presence | 75 | 79 | 79 |
| vowel_model | MHG | 5,984 | 29,554 | document-lemma-slot-outcome presence | 151 | 124 | 124 |

## Normalization exclusions

Frequency attrition is separate from missing metadata: **588**
mapped tokens belong to lemma families with 10 or fewer mapped tokens. The next
table uses the 280,870 frequency-eligible tokens as its
denominator. Counts are independent and may overlap.

| criterion | denominator | excluded_tokens | definition |
| :--- | :--- | :--- | :--- |
| missing date mapping | 280,870 | 127,410 | independent count; overlaps other missing-field counts |
| missing variety mapping | 280,870 | 9,937 | independent count; overlaps other missing-field counts |
| missing principal-part mapping | 280,870 | 8,276 | independent count; overlaps other missing-field counts |

The mutually exclusive audit applies the listed reasons in date, variety,
principal-part order and therefore sums to the normalized total:

| exclusive_reason | excluded_tokens |
| :--- | :--- |
| missing date mapping | 127,410 |
| missing variety mapping | 7,612 |
| missing principal-part mapping | 3,803 |

## Baseline anchors

There are **954** supported anchor cells; **160**
rest on one token. Support is tabulated over all three study slots in
`baseline_anchor_support.csv`; the binned distribution is:

| std_infl | support_bin | anchor_cells | denominator |
| :--- | :--- | :--- | :--- |
| PastPl | 1 | 54 | 954 |
| PastPl | 2 | 28 | 954 |
| PastPl | 3-4 | 47 | 954 |
| PastPl | 5+ | 151 | 954 |
| PastSg | 1 | 52 | 954 |
| PastSg | 2 | 29 | 954 |
| PastSg | 3-4 | 33 | 954 |
| PastSg | 5+ | 206 | 954 |
| Pres | 1 | 54 | 954 |
| Pres | 2 | 32 | 954 |
| Pres | 3-4 | 36 | 954 |
| Pres | 5+ | 232 | 954 |

## Target sources

The denominator is all **531 lemma-variety groups** in the
normalized corpus. Missing target rows remain in the table rather than being
silently removed.

| tense | source | groups | denominator_all_lemma_variety_groups |
| :--- | :--- | :--- | :--- |
| pres | nhg | 406 | 531 |
| pres | missing target row | 125 | 531 |
| pres | resolved vowel target (all sources) | 406 | 531 |
| past | nhg | 397 | 531 |
| past | missing target row | 125 | 531 |
| past | corpus | 6 | 531 |
| past | none | 3 | 531 |
| past | resolved vowel target (all sources) | 400 | 531 |

## Outcomes

| marking_type | observations | leveled | preserved | leveling_pct |
| :--- | :--- | :--- | :--- | :--- |
| vowel_unipartite | 35,413 | 351 | 35,062 | 0.9912 |
| vowel_bipartite | 2,878 | 17 | 2,861 | 0.5907 |
| consonant_bipartite | 1,212 | 68 | 1,144 | 5.611 |

## Production-aligned sound-change exclusions

The production coder protects contrasts already present in each paradigm's
baseline. With that protection, **528** past-target and
**15** present-target comparisons are made
uninformative by a regular sound-change equivalence. Calling the vowel helper
without protected contrasts would instead report
**2,508**;
that is a counterfactual diagnostic and is not the production exclusion count.

## Vowel model support

The model table contains **7,510 distinct document-lemma-slot-outcome
observations** from **7,422** cells representing
**38,291** source-token outcomes. Repetition
counts remain audit columns and do not enter the Bernoulli likelihood.
**124** lemma families, and
**226** real source documents. Predictor levels
and numeric ranges are in `model_predictor_distribution.csv`.

*Generated by `analysis/attrition_diagnostics.py`.*
