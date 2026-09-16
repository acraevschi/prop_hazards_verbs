#!/usr/bin/env python3
"""Compare production targets with late-corpus target definitions."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.corpus_approach_coding import (
    step_1_preprocessing,
    step_2_establish_baseline,
    step_3_establish_targets,
    step_4_coding_outcome,
)


TARGET_COLUMNS = [
    "target_vowel_pres", "target_coda_pres",
    "target_vowel_past", "target_coda_past",
]


def _mode_or_na(values):
    mode = values.dropna().mode()
    return mode.iloc[0] if not mode.empty else pd.NA


def _target_rows_for_tense(group, slots, threshold_date, fallback):
    rows = group[group["std_infl"].isin(slots)]
    late = rows[rows["date_num"] >= threshold_date]
    if not late.empty:
        return late
    if not fallback or rows.empty:
        return rows.iloc[0:0]
    latest = rows["date_num"].max()
    return rows[rows["date_num"] == latest]


def compute_corpus_targets(enhg_df, threshold_date=1500, fallback=False):
    """Build corpus targets from date >= cutoff, optionally falling back per tense."""
    rows = []
    for (lemma_id, variety), group in enhg_df.groupby(["lemma_id", "variety"]):
        out = {"lemma_id": lemma_id, "variety": variety}
        for tense, slots in (("pres", ["Pres"]), ("past", ["PastSg", "PastPl"])):
            chosen = _target_rows_for_tense(group, slots, threshold_date, fallback)
            out[f"target_vowel_{tense}"] = _mode_or_na(chosen["extracted_vowel"])
            out[f"target_coda_{tense}"] = _mode_or_na(chosen["extracted_coda"])
            out[f"target_{tense}_source"] = (
                "corpus >= cutoff" if not chosen.empty and (chosen["date_num"] >= threshold_date).all()
                else "latest corpus fallback" if not chosen.empty else "missing"
            )
        rows.append(out)
    return pd.DataFrame(rows)


def compute_targets_strict(enhg_df, threshold_date=1500):
    return compute_corpus_targets(enhg_df, threshold_date, fallback=False)


def compute_targets_hybrid(enhg_df, threshold_date=1500):
    return compute_corpus_targets(enhg_df, threshold_date, fallback=True)


def _complete_and_overlay_nhg(universe, production, alternative):
    """Hold curated modern targets fixed so the comparison isolates corpus dates."""
    keys = ["lemma_id", "variety"]
    base = universe.merge(production, on=keys, how="left", validate="one_to_one")
    alt = universe.merge(alternative, on=keys, how="left", validate="one_to_one")
    for tense in ("pres", "past"):
        modern = base[f"target_{tense}_source"].eq("nhg")
        for part in ("vowel", "coda"):
            column = f"target_{part}_{tense}"
            alt.loc[modern, column] = base.loc[modern, column]
        alt.loc[modern, f"target_{tense}_source"] = "nhg"
    return base, alt


def extract_has_levelled(df):
    """Return one row per observation with the same OR collapse as run_brms.R."""
    def collapse(first, second):
        a = pd.to_numeric(df[first], errors="coerce")
        b = pd.to_numeric(df[second], errors="coerce")
        return np.where((a == 1) | (b == 1), 1,
                        np.where((a == 0) | (b == 0), 0, np.nan))

    return pd.DataFrame(
        {
            "observation_id": df["observation_id"],
            "Vowels": collapse("is_leveled_vowel_pres", "is_leveled_vowel_past"),
            "Consonants": collapse("is_leveled_cons_pres", "is_leveled_cons_past"),
        }
    )


def _outcome_rows(production_coded, alternative_coded, comparison):
    left = extract_has_levelled(production_coded)
    right = extract_has_levelled(alternative_coded)
    merged = left.merge(right, on="observation_id", how="outer", validate="one_to_one",
                        suffixes=("_production", "_alternative"))
    rows = []
    for element in ("Vowels", "Consonants"):
        base = merged[f"{element}_production"]
        alt = merged[f"{element}_alternative"]
        base_ok, alt_ok = base.notna(), alt.notna()
        both = base_ok & alt_ok
        concordant = int((base[both] == alt[both]).sum())
        discordant = int((base[both] != alt[both]).sum())
        rows.append(
            {
                "comparison": comparison,
                "element": element,
                "all_coded_rows": len(merged),
                "production_codable": int(base_ok.sum()),
                "alternative_codable": int(alt_ok.sum()),
                "codable_in_both": int(both.sum()),
                "production_only": int((base_ok & ~alt_ok).sum()),
                "alternative_only": int((~base_ok & alt_ok).sum()),
                "neither_codable": int((~base_ok & ~alt_ok).sum()),
                "concordant_labels": concordant,
                "discordant_labels": discordant,
                "flips_0_to_1": int(((base[both] == 0) & (alt[both] == 1)).sum()),
                "flips_1_to_0": int(((base[both] == 1) & (alt[both] == 0)).sum()),
                "outcome_agreement_pct_among_both": 100 * concordant / both.sum() if both.sum() else np.nan,
            }
        )
    return rows


def _target_rows(base, alt, comparison):
    rows = []
    for column in TARGET_COLUMNS:
        production = base[column]
        alternative = alt[column]
        base_ok, alt_ok = production.notna(), alternative.notna()
        both = base_ok & alt_ok
        same = int((production[both] == alternative[both]).sum())
        rows.append(
            {
                "comparison": comparison,
                "target_component": column,
                "all_lemma_variety_groups": len(base),
                "production_resolved": int(base_ok.sum()),
                "alternative_resolved": int(alt_ok.sum()),
                "resolved_in_both": int(both.sum()),
                "production_only": int((base_ok & ~alt_ok).sum()),
                "alternative_only": int((~base_ok & alt_ok).sum()),
                "neither_resolved": int((~base_ok & ~alt_ok).sum()),
                "identical_targets": same,
                "different_targets": int((production[both] != alternative[both]).sum()),
                "direct_target_agreement_pct_among_both": 100 * same / both.sum() if both.sum() else np.nan,
            }
        )
    return rows


def _markdown(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(":---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (int, np.integer)):
                values.append(f"{value:,}")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}" if pd.notna(value) else "NA")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_target_sensitivity(
    norm_file="data/combined_normalized_corpus.csv",
    sc_file="data/vowel_changes.csv",
    output_report="analysis/reports/target_sensitivity_report.md",
    output_csv="analysis/reports/target_sensitivity_summary.csv",
    target_csv="analysis/reports/target_sensitivity_targets.csv",
    threshold_date=1500,
):
    print("=" * 70)
    print("Running target-definition sensitivity analysis...")
    print("=" * 70)

    normalized = pd.read_csv(norm_file, dtype=str)
    processed = step_1_preprocessing(normalized)
    baseline = step_2_establish_baseline(processed)
    production = step_3_establish_targets(processed)
    enhg = processed[processed["corpus"] == "ENHG"].copy()
    enhg["date_num"] = pd.to_numeric(enhg["date"], errors="coerce")
    universe = processed[["lemma_id", "variety"]].dropna().drop_duplicates()

    alternatives = {
        f"strict corpus date >= {threshold_date}": compute_targets_strict(enhg, threshold_date),
        f"late corpus with per-tense fallback": compute_targets_hybrid(enhg, threshold_date),
    }
    production_coded = step_4_coding_outcome(processed, baseline, production, sc_file)
    outcome_rows = []
    target_rows = []
    for label, alternative in alternatives.items():
        base, alt = _complete_and_overlay_nhg(universe, production, alternative)
        alternative_coded = step_4_coding_outcome(processed, baseline, alt, sc_file)
        outcome_rows.extend(_outcome_rows(production_coded, alternative_coded, label))
        target_rows.extend(_target_rows(base, alt, label))

    outcomes = pd.DataFrame(outcome_rows)
    targets = pd.DataFrame(target_rows)
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    outcomes.to_csv(output_csv, index=False)
    targets.to_csv(target_csv, index=False)

    report = f"""# Target-Definition Sensitivity Audit

The production endpoint uses a curated modern German form where available and
the latest usable ENHG date as fallback. Both alternatives keep curated modern
targets fixed and change only the corpus fallback: either require corpus rows
dated at least {threshold_date}, or use those rows when available and fall back
per tense to the latest corpus date.

Outcome agreement and direct target identity are different quantities. Outcome
agreement is reported only among observations codable under both definitions;
the production-only, alternative-only, and neither-codable counts keep that
conditional denominator visible.

## Outcome coding

{_markdown(outcomes)}

## Direct target identity

The denominator here is every lemma-variety group in the normalized corpus,
including groups where one or both definitions do not resolve the component.

{_markdown(targets)}

No percentage in either table is presented as agreement over the missing rows.

*Generated by `analysis/target_sensitivity.py`.*
"""
    with open(output_report, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Wrote {output_report}, {output_csv}, and {target_csv}")
    print(outcomes.to_string(index=False))
    return outcomes


if __name__ == "__main__":
    run_target_sensitivity()
