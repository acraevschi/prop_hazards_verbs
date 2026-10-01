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
| Normalized | 280,870 | 211,915 | 211,915 | token row | 2,659 | 291 | 349 | date, variety, and principal part all resolved |
| Coding eligible | 211,915 | 151,045 | 151,045 | token row | 2,393 | 285 | 349 | excluded non-strong/irregular inflClass labels |
| Coded past tokens | 151,045 | 56,498 | 56,498 | token row | 1,646 | 245 | 338 | PastSg and PastPl rows emitted by outcome coder |
| Vowel model | 56,498 | 9,290 | 45,390 | document-lemma-slot-outcome presence | 126 | 126 | 335 | distinct document x lemma-family x slot x outcome rows |

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
| normalized | ENHG | 116,065 | 116,065 | token row | 188 | 459 | 198 |
| normalized | MHG | 95,850 | 95,850 | token row | 161 | 2,221 | 227 |
| coding_eligible | ENHG | 72,053 | 72,053 | token row | 188 | 445 | 197 |
| coding_eligible | MHG | 78,992 | 78,992 | token row | 161 | 1,968 | 216 |
| coded | ENHG | 20,196 | 20,196 | token row | 182 | 309 | 143 |
| coded | MHG | 36,302 | 36,302 | token row | 156 | 1,349 | 197 |
| vowel_model | ENHG | 3,166 | 15,414 | document-lemma-slot-outcome presence | 179 | 83 | 83 |
| vowel_model | MHG | 6,124 | 29,976 | document-lemma-slot-outcome presence | 156 | 126 | 126 |

## Normalization exclusions

Frequency attrition is separate from missing metadata: **588**
mapped tokens belong to lemma families with 10 or fewer mapped tokens. The next
table uses the 280,870 frequency-eligible tokens as its
denominator. Counts are independent and may overlap.

| criterion | denominator | excluded_tokens | definition |
| :--- | :--- | :--- | :--- |
| missing date mapping | 280,870 | 57,349 | independent count; overlaps other missing-field counts |
| missing variety mapping | 280,870 | 9,937 | independent count; overlaps other missing-field counts |
| missing principal-part mapping | 280,870 | 8,276 | independent count; overlaps other missing-field counts |

The mutually exclusive audit applies the listed reasons in date, variety,
principal-part order and therefore sums to the normalized total:

| exclusive_reason | excluded_tokens |
| :--- | :--- |
| missing date mapping | 57,349 |
| missing variety mapping | 7,612 |
| missing principal-part mapping | 3,994 |

## Baseline anchors

There are **960** supported anchor cells; **162**
rest on one token. Support is tabulated over all three study slots in
`baseline_anchor_support.csv`; the binned distribution is:

| std_infl | support_bin | anchor_cells | denominator |
| :--- | :--- | :--- | :--- |
| PastPl | 1 | 53 | 960 |
| PastPl | 2 | 31 | 960 |
| PastPl | 3-4 | 44 | 960 |
| PastPl | 5+ | 154 | 960 |
| PastSg | 1 | 55 | 960 |
| PastSg | 2 | 29 | 960 |
| PastSg | 3-4 | 33 | 960 |
| PastSg | 5+ | 206 | 960 |
| Pres | 1 | 54 | 960 |
| Pres | 2 | 33 | 960 |
| Pres | 3-4 | 35 | 960 |
| Pres | 5+ | 233 | 960 |

## Target sources

The denominator is all **548 lemma-variety groups** in the
normalized corpus. Missing target rows remain in the table rather than being
silently removed.

| tense | source | groups | denominator_all_lemma_variety_groups |
| :--- | :--- | :--- | :--- |
| pres | nhg | 422 | 548 |
| pres | missing target row | 126 | 548 |
| pres | resolved vowel target (all sources) | 422 | 548 |
| past | nhg | 413 | 548 |
| past | missing target row | 126 | 548 |
| past | corpus | 7 | 548 |
| past | none | 2 | 548 |
| past | resolved vowel target (all sources) | 417 | 548 |

## Outcomes

| marking_type | observations | leveled | preserved | leveling_pct |
| :--- | :--- | :--- | :--- | :--- |
| vowel_unipartite | 42,028 | 576 | 41,452 | 1.371 |
| vowel_bipartite | 3,362 | 57 | 3,305 | 1.695 |
| consonant_bipartite | 1,466 | 155 | 1,311 | 10.57 |

## Production-aligned sound-change exclusions

The production coder protects contrasts already present in each paradigm's
baseline. With that protection, **603** past-target and
**19** present-target comparisons are made
uninformative by a regular sound-change equivalence. Calling the vowel helper
without protected contrasts would instead report
**2,753**;
that is a counterfactual diagnostic and is not the production exclusion count.

## Vowel model support

The model table contains **9,290 distinct document-lemma-slot-outcome
observations** from **9,174** cells representing
**45,390** source-token outcomes. Repetition
counts remain audit columns and do not enter the Bernoulli likelihood.
**126** lemma families, and
**335** real source documents. Predictor levels
and numeric ranges are in `model_predictor_distribution.csv`.

*Generated by `analysis/attrition_diagnostics.py`.*
