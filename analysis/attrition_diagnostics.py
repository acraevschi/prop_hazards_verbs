#!/usr/bin/env python3
"""Audit every data-reduction step used by the production analysis."""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.marking_type_summary import reshape
from data.corpus_approach_coding import (
    BASELINE_MAX_DATE,
    BASELINE_SLOTS,
    are_vowels_equivalent,
    baseline_contrast_context,
    load_sound_changes,
    step_1_preprocessing,
    step_2_establish_baseline,
    step_3_establish_targets,
)
from data.normalize_data import map_category


REPORT_DIR = "analysis/reports"


def _read_required(path, **kwargs):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required pipeline artifact is missing: {path}")
    return pd.read_csv(path, **kwargs)


def _nunique(frame, column):
    return int(frame[column].nunique()) if column in frame else 0


def _stage_row(stage, denominator, frame, note, source_tokens=None, unit="token row"):
    return {
        "stage": stage,
        "denominator": denominator,
        "observations": len(frame),
        "source_tokens": len(frame) if source_tokens is None else int(source_tokens),
        "observation_unit": unit,
        "surface_lemmas": _nunique(frame, "lemma"),
        "lemma_families": _nunique(frame, "lemma_id"),
        "documents": _nunique(frame, "document_id"),
        "note": note,
    }


def _markdown_table(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(":---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (int, np.integer)):
                values.append(f"{value:,}")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _normalization_audit(combined, dialect_file, date_file):
    """Reproduce normalization gates without changing the production artifact."""
    with open(dialect_file, encoding="utf-8") as handle:
        dialect = {x["original"]: x["normalized"] for x in json.load(handle)}
    with open(date_file, encoding="utf-8") as handle:
        dates = {x["original"]: x["normalized"] for x in json.load(handle)}

    work = combined.copy()
    work["mapped"] = work["lemma_id"].notna()
    family_size = work.loc[work["mapped"]].groupby("lemma_id")["lemma_id"].transform("size")
    work.loc[work["mapped"], "family_size"] = family_size
    work["passes_frequency"] = work["mapped"] & work["family_size"].gt(10)
    eligible = work.loc[work["passes_frequency"]].copy()
    eligible["mapped_date"] = eligible["date"].map(dates)
    eligible["mapped_variety"] = eligible["language-region"].map(dialect)
    eligible["mapped_principal_part"] = eligible["infl"].apply(map_category)

    failures = pd.DataFrame(
        [
            {
                "criterion": "missing date mapping",
                "denominator": len(eligible),
                "excluded_tokens": int(eligible["mapped_date"].isna().sum()),
                "definition": "independent count; overlaps other missing-field counts",
            },
            {
                "criterion": "missing variety mapping",
                "denominator": len(eligible),
                "excluded_tokens": int(eligible["mapped_variety"].isna().sum()),
                "definition": "independent count; overlaps other missing-field counts",
            },
            {
                "criterion": "missing principal-part mapping",
                "denominator": len(eligible),
                "excluded_tokens": int(eligible["mapped_principal_part"].isna().sum()),
                "definition": "independent count; overlaps other missing-field counts",
            },
        ]
    )

    remaining = pd.Series(True, index=eligible.index)
    exclusive = []
    for label, column in (
        ("missing date mapping", "mapped_date"),
        ("missing variety mapping", "mapped_variety"),
        ("missing principal-part mapping", "mapped_principal_part"),
    ):
        mask = remaining & eligible[column].isna()
        exclusive.append({"exclusive_reason": label, "excluded_tokens": int(mask.sum())})
        remaining &= ~mask

    return work, eligible, failures, pd.DataFrame(exclusive), eligible.loc[remaining]


def _coerce_bool(value):
    if value is True or str(value).strip().lower() == "true":
        return True
    if value is False or str(value).strip().lower() == "false":
        return False
    return pd.NA


def _sound_change_audit(coded, sc_dict):
    """Count the target-invalidity rule exactly as production applies it."""
    flag_cols = [
        "diff_vowel_pres_pastsg",
        "diff_vowel_pres_pastpl",
        "diff_vowel_pastsg_pastpl",
        "diff_cons_pres_pastsg",
        "diff_cons_pres_pastpl",
        "diff_cons_pastsg_pastpl",
    ]
    work = coded.copy()
    for column in flag_cols:
        work[column] = work[column].map(_coerce_bool)

    counts = {
        "past_protected": 0,
        "present_protected": 0,
        "past_unprotected_counterfactual": 0,
        "present_unprotected_counterfactual": 0,
    }
    for _, row in work[work["is_bipartite"].notna()].iterrows():
        infl = row.get("std_infl")
        if infl not in ("PastSg", "PastPl"):
            continue
        context = baseline_contrast_context(row, infl)
        anchor_self = context["anchor_self_v"]
        if pd.isna(anchor_self):
            continue
        for label, target_col, compare_key, diff_key in (
            ("past", "target_vowel_past", "anchor_other_v", "hist_diff_v_other"),
            ("present", "target_vowel_pres", "anchor_pres_v", "hist_diff_v_pres"),
        ):
            target = row.get(target_col)
            if context[diff_key] is not True or pd.isna(context[compare_key]) or pd.isna(target):
                continue
            if str(target) == str(anchor_self):
                continue
            if are_vowels_equivalent(target, anchor_self, row.get("variety", ""), sc_dict):
                counts[f"{label}_unprotected_counterfactual"] += 1
            if are_vowels_equivalent(
                target,
                anchor_self,
                row.get("variety", ""),
                sc_dict,
                protected=context["protected_v"],
            ):
                counts[f"{label}_protected"] += 1
    return counts


def run_attrition_diagnostics(
    mhg_file="data/mhg_corpus.csv",
    enhg_file="data/enhg_corpus.csv",
    comb_file="data/combined_corpus.csv",
    norm_file="data/combined_normalized_corpus.csv",
    coded_file="data/coded_output.csv",
    analysis_file="analysis/data_for_analysis.csv",
    sc_file="data/vowel_changes.csv",
    dialect_file="data/dialect_mapping.json",
    date_file="data/date_mapping.json",
    output_report=f"{REPORT_DIR}/attrition_report.md",
    output_csv=f"{REPORT_DIR}/attrition_summary.csv",
):
    print("=" * 70)
    print("Running production-aligned attrition diagnostics...")
    print("=" * 70)

    mhg = _read_required(mhg_file, dtype=str)
    enhg = _read_required(enhg_file, dtype=str)
    combined = _read_required(comb_file, dtype=str)
    normalized = _read_required(norm_file, dtype=str)
    coded = _read_required(coded_file, dtype=str)
    model = _read_required(analysis_file)
    long = reshape(coded_file)

    identity_cols = ["document_id", "token_id", "observation_id"]
    for path, frame in ((mhg_file, mhg), (enhg_file, enhg), (comb_file, combined),
                        (norm_file, normalized), (coded_file, coded)):
        missing = set(identity_cols) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing identity columns: {sorted(missing)}")
        if frame[identity_cols].isna().any().any():
            raise ValueError(f"{path} has missing identity values")
        if frame["observation_id"].duplicated().any():
            raise ValueError(f"{path} has repeated observation_id values")
    required_model = {"model_row_id", "cell_id", "document_id", "lemma_std",
                      "std_infl", "has_levelled", "leveled_tokens",
                      "preserved_tokens", "n_tokens"}
    missing_model = required_model - set(model.columns)
    if missing_model:
        raise ValueError(f"{analysis_file} is missing cell columns: {sorted(missing_model)}")
    if model["model_row_id"].duplicated().any() or model.duplicated(
        ["document_id", "lemma_std", "std_infl", "has_levelled"]
    ).any():
        raise ValueError(f"{analysis_file} has repeated cell-outcome observations")
    if not (
        model["leveled_tokens"] + model["preserved_tokens"] == model["n_tokens"]
    ).all():
        raise ValueError(f"{analysis_file} has inconsistent audit token counts")

    norm_work, threshold_eligible, missing_fields, exclusive, reconstructed_norm = (
        _normalization_audit(combined, dialect_file, date_file)
    )
    if set(reconstructed_norm["observation_id"]) != set(normalized["observation_id"]):
        raise ValueError("Normalization audit does not reproduce combined_normalized_corpus.csv")

    mapped = combined[combined["lemma_id"].notna()]
    frequency_kept = norm_work[norm_work["passes_frequency"]]
    model_with_family = model.rename(columns={"lemma_std": "lemma_id"})
    model_cells = model.drop_duplicates("cell_id")
    processed = step_1_preprocessing(normalized)
    stages = pd.DataFrame(
        [
            _stage_row("MHG extraction", len(mhg), mhg, "all extracted MHG strong-verb tokens"),
            _stage_row("ENHG extraction", len(enhg), enhg, "all extracted ENHG strong-verb tokens"),
            _stage_row("Combined join", len(mhg) + len(enhg), combined, "mapped and unmapped rows retained"),
            _stage_row("Mapped join rows", len(combined), mapped, "non-missing corpus-aware lemma_id"),
            _stage_row("Frequency eligible", len(mapped), frequency_kept, "family has more than 10 mapped tokens"),
            _stage_row("Normalized", len(frequency_kept), normalized, "date, variety, and principal part all resolved"),
            _stage_row("Coding eligible", len(normalized), processed, "excluded non-strong/irregular inflClass labels"),
            _stage_row("Coded past tokens", len(processed), coded, "PastSg and PastPl rows emitted by outcome coder"),
            _stage_row(
                "Vowel model", len(coded), model_with_family,
                "distinct document x lemma-family x slot x outcome rows",
                source_tokens=model_cells["n_tokens"].sum(),
                unit="document-lemma-slot-outcome presence",
            ),
        ]
    )

    corpus_rows = []
    extracted = pd.concat(
        [mhg.assign(corpus="MHG"), enhg.assign(corpus="ENHG")],
        ignore_index=True,
    )
    for stage, frame in (
        ("extracted", extracted),
        ("combined", combined),
        ("mapped", mapped),
        ("normalized", normalized),
        ("coding_eligible", processed),
        ("coded", coded),
        ("vowel_model", model_with_family),
    ):
        for corpus, group in frame.groupby("corpus", dropna=False):
            observations = len(group)
            source_tokens = (
                int(group.drop_duplicates("cell_id")["n_tokens"].sum())
                if stage == "vowel_model" else len(group)
            )
            corpus_rows.append(
                {
                    "stage": stage,
                    "corpus": corpus,
                    "observations": observations,
                    "source_tokens": source_tokens,
                    "observation_unit": "document-lemma-slot-outcome presence" if stage == "vowel_model" else "token row",
                    "documents": _nunique(group, "document_id"),
                    "surface_lemmas": _nunique(group, "lemma"),
                    "lemma_families": _nunique(group, "lemma_id"),
                }
            )
    corpus_counts = pd.DataFrame(corpus_rows)

    baseline_candidates = processed[
        (processed["corpus"] == "MHG")
        & (processed["date"] <= BASELINE_MAX_DATE)
        & (processed["std_infl"].isin(BASELINE_SLOTS))
        & processed["extracted_vowel"].notna()
    ]
    def mode_support(values):
        counts = values.dropna().value_counts()
        return int(counts.iloc[0]) if len(counts) else 0

    def mode_tied(values):
        counts = values.dropna().value_counts()
        return bool(len(counts) > 1 and counts.iloc[0] == counts.iloc[1])

    anchor_support = (
        baseline_candidates.groupby(["lemma_id", "variety", "std_infl"])
        .agg(
            support_tokens=("extracted_vowel", "size"),
            vowel_mode_support=("extracted_vowel", mode_support),
            vowel_mode_tied=("extracted_vowel", mode_tied),
            coda_mode_support=("extracted_coda", mode_support),
            coda_mode_tied=("extracted_coda", mode_tied),
        )
        .reset_index()
    )
    baseline = step_2_establish_baseline(processed)
    expected_anchor_cells = sum(
        baseline[f"anchor_vowel_{slot.lower()}"].notna().sum()
        for slot in BASELINE_SLOTS
    )
    if expected_anchor_cells != len(anchor_support):
        raise ValueError("Baseline support cells disagree with production anchors")
    support_distribution = (
        anchor_support.assign(
            support_bin=pd.cut(
                anchor_support["support_tokens"],
                bins=[0, 1, 2, 4, np.inf],
                labels=["1", "2", "3-4", "5+"],
            )
        )
        .groupby(["std_infl", "support_bin"], observed=False)
        .size().rename("anchor_cells").reset_index()
    )
    support_distribution["denominator"] = len(anchor_support)

    targets = step_3_establish_targets(processed)
    target_universe = processed[["lemma_id", "variety"]].drop_duplicates()
    target_audit = target_universe.merge(targets, on=["lemma_id", "variety"], how="left")
    target_source_rows = []
    for tense in ("pres", "past"):
        source = target_audit[f"target_{tense}_source"].fillna("missing target row")
        for name, count in source.value_counts(dropna=False).items():
            target_source_rows.append(
                {
                    "tense": tense,
                    "source": name,
                    "groups": int(count),
                    "denominator_all_lemma_variety_groups": len(target_universe),
                }
            )
        resolved = target_audit[f"target_vowel_{tense}"].notna()
        target_source_rows.append(
            {
                "tense": tense,
                "source": "resolved vowel target (all sources)",
                "groups": int(resolved.sum()),
                "denominator_all_lemma_variety_groups": len(target_universe),
            }
        )
    target_sources = pd.DataFrame(target_source_rows)

    channel_rows = []
    for marking in ("vowel_unipartite", "vowel_bipartite", "consonant_bipartite"):
        subset = long[long["marking_type"] == marking]
        channel_rows.append(
            {
                "marking_type": marking,
                "observations": len(subset),
                "leveled": int(subset["has_levelled"].sum()),
                "preserved": int((subset["has_levelled"] == 0).sum()),
                "leveling_pct": 100 * subset["has_levelled"].mean() if len(subset) else 0.0,
            }
        )
    channel_outcomes = pd.DataFrame(channel_rows)

    sound = _sound_change_audit(coded, load_sound_changes(sc_file))

    predictor_rows = []
    for column in ("date", "log_freq", "log_token_freq", "log_alt_pres_freq", "log_alt_past_freq",
                   "n_tokens", "leveled_tokens", "preserved_tokens", "n_log_freq_values"):
        values = pd.to_numeric(model[column], errors="coerce")
        predictor_rows.append(
            {
                "predictor": column,
                "level": "numeric summary",
                "n": int(values.notna().sum()),
                "mean": values.mean(),
                "sd": values.std(),
                "min": values.min(),
                "median": values.median(),
                "max": values.max(),
            }
        )
    for column in ("marking_type", "std_infl", "variety", "corpus", "has_alt_pres", "has_alt_past"):
        for level, count in model[column].value_counts(dropna=False).items():
            predictor_rows.append(
                {"predictor": column, "level": level, "n": int(count),
                 "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan}
            )
    for column in ("lemma_std", "document_id"):
        support = model.groupby(column)["n_tokens"].sum()
        predictor_rows.append(
            {
                "predictor": column,
                "level": "source-token support per grouping level",
                "n": len(support),
                "mean": support.mean(),
                "sd": support.std(),
                "min": support.min(),
                "median": support.median(),
                "max": support.max(),
            }
        )
    predictors = pd.DataFrame(predictor_rows)

    mapping_rows = []
    for corpus, group in combined.groupby("corpus"):
        mapping_rows.append(
            {
                "corpus": corpus,
                "joined_rows": len(group),
                "mapped_rows": int(group["lemma_id"].notna().sum()),
                "unmapped_rows": int(group["lemma_id"].isna().sum()),
                "denominator": len(group),
            }
        )
    mapping = pd.DataFrame(mapping_rows)

    summary_rows = [
        ("join", "joined rows", len(combined)),
        ("join", "mapped rows", len(mapped)),
        ("join", "unmapped rows", int(combined["lemma_id"].isna().sum())),
        ("normalization", "mapped tokens in families with <=10 tokens", int((norm_work["mapped"] & ~norm_work["passes_frequency"]).sum())),
        ("normalization", "tokens after frequency threshold", len(threshold_eligible)),
        ("normalization", "normalized tokens", len(normalized)),
        ("baseline", "anchor cells in Pres PastSg PastPl", len(anchor_support)),
        ("baseline", "one-token anchor cells", int((anchor_support["support_tokens"] == 1).sum())),
        ("baseline", "anchored lemma-variety paradigms", len(baseline)),
        ("coding", "coded past tokens", len(coded)),
        ("model", "document-lemma-slot-outcome observations", len(model)),
        ("model", "document-lemma-slot cells", model["cell_id"].nunique()),
        ("model", "vowel token outcomes represented", int(model_cells["n_tokens"].sum())),
        ("model", "vowel modeling lemmas", _nunique(model, "lemma_std")),
        ("model", "vowel modeling documents", _nunique(model, "document_id")),
        ("sound change", "protected past target-anchor exclusions", sound["past_protected"]),
        ("sound change", "protected present target-anchor exclusions", sound["present_protected"]),
        ("sound change", "unprotected counterfactual exclusions", sound["past_unprotected_counterfactual"] + sound["present_unprotected_counterfactual"]),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric_category", "metric", "value"])

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    outputs = {
        output_csv: summary,
        f"{REPORT_DIR}/stage_counts.csv": stages,
        f"{REPORT_DIR}/corpus_stage_counts.csv": corpus_counts,
        f"{REPORT_DIR}/normalization_exclusions.csv": missing_fields,
        f"{REPORT_DIR}/normalization_exclusions_exclusive.csv": exclusive,
        f"{REPORT_DIR}/lemma_join_counts.csv": mapping,
        f"{REPORT_DIR}/baseline_anchor_support.csv": anchor_support,
        f"{REPORT_DIR}/baseline_support_distribution.csv": support_distribution,
        f"{REPORT_DIR}/target_source_counts.csv": target_sources,
        f"{REPORT_DIR}/channel_outcomes.csv": channel_outcomes,
        f"{REPORT_DIR}/model_predictor_distribution.csv": predictors,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)

    report = f"""# Attrition and Data-Support Audit

All counts below were regenerated from the pipeline artifacts. The baseline is
inclusive: MHG observations dated **1200 or earlier** (`date <= {BASELINE_MAX_DATE}`),
and it uses exactly `{', '.join(BASELINE_SLOTS)}`. Participles are not baseline
anchors because they enter neither production contrast.

## Stage counts

`denominator` names the immediately relevant universe for each row; the notes
state the gate. MHG and ENHG extraction rows are separate and are reconciled by
the combined-join row.

{_markdown_table(stages)}

## Join audit by corpus

{_markdown_table(mapping)}

## Corpus and document support

{_markdown_table(corpus_counts)}

## Normalization exclusions

Frequency attrition is separate from missing metadata: **{int((norm_work['mapped'] & ~norm_work['passes_frequency']).sum()):,}**
mapped tokens belong to lemma families with 10 or fewer mapped tokens. The next
table uses the {len(threshold_eligible):,} frequency-eligible tokens as its
denominator. Counts are independent and may overlap.

{_markdown_table(missing_fields)}

The mutually exclusive audit applies the listed reasons in date, variety,
principal-part order and therefore sums to the normalized total:

{_markdown_table(exclusive)}

## Baseline anchors

There are **{len(anchor_support):,}** supported anchor cells; **{int((anchor_support['support_tokens'] == 1).sum()):,}**
rest on one token. Support is tabulated over all three study slots in
`baseline_anchor_support.csv`; the binned distribution is:

{_markdown_table(support_distribution)}

## Target sources

The denominator is all **{len(target_universe):,} lemma-variety groups** in the
normalized corpus. Missing target rows remain in the table rather than being
silently removed.

{_markdown_table(target_sources)}

## Outcomes

{_markdown_table(channel_outcomes)}

## Production-aligned sound-change exclusions

The production coder protects contrasts already present in each paradigm's
baseline. With that protection, **{sound['past_protected']:,}** past-target and
**{sound['present_protected']:,}** present-target comparisons are made
uninformative by a regular sound-change equivalence. Calling the vowel helper
without protected contrasts would instead report
**{sound['past_unprotected_counterfactual'] + sound['present_unprotected_counterfactual']:,}**;
that is a counterfactual diagnostic and is not the production exclusion count.

## Vowel model support

The model table contains **{len(model):,} distinct document-lemma-slot-outcome
observations** from **{model['cell_id'].nunique():,}** cells representing
**{int(model_cells['n_tokens'].sum()):,}** source-token outcomes. Repetition
counts remain audit columns and do not enter the Bernoulli likelihood.
**{_nunique(model, 'lemma_std'):,}** lemma families, and
**{_nunique(model, 'document_id'):,}** real source documents. Predictor levels
and numeric ranges are in `model_predictor_distribution.csv`.

*Generated by `analysis/attrition_diagnostics.py`.*
"""
    with open(output_report, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"Wrote {output_report} and {len(outputs)} machine-readable audit tables.")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_attrition_diagnostics()
