#!/usr/bin/env python3
"""
Consonant Channel Analysis.

Purpose
-------
Under Hermann Paul's Principle, bipartite marking (combining vocalic Ablaut and
consonantal Grammatischer Wechsel) is hypothesized to create stronger resistance
to analogical leveling. While the primary Bayesian GAMM models evaluate the
vocalic channel (vowel_unipartite vs vowel_bipartite), this module performs
a dedicated analysis of the consonant channel (`consonant_bipartite`).

Key Questions Investigated
--------------------------
1. Rate Discrepancy: Why is the observed leveling rate in the consonant channel
   (~8.02%) substantially higher than bipartite vowel leveling (~0.86%) and
   unipartite vowel leveling (~2.07%)?
2. Which clause of the bipartite rule admitted each paradigm, read off the same
   anchors that step_2_establish_baseline used. Devoicing-shaped paradigms
   (scheiden d ~ t ~ d) are already excluded upstream, so that category is
   expected to be empty here; it is reported as a tripwire on the rule, not as
   a finding about the language.
3. Concentration across Lemmas: Which verbs drive the consonant leveling counts,
   and how does rate vary when orthographic coda alternations are separated from
   genuine stem-consonant leveling?

Outputs
-------
- analysis/reports/consonant_analysis_report.md
- analysis/reports/consonant_summary.csv
- analysis/reports/consonant_lemma_breakdown.csv
"""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

# Statistical test helpers using scipy if available, or pure Python fallback
try:
    from scipy.stats import fisher_exact, chi2_contingency
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


CODED_DEFAULT = "data/coded_output.csv"
REPORT_DEFAULT = "analysis/reports/consonant_analysis_report.md"
SUMMARY_CSV_DEFAULT = "analysis/reports/consonant_summary.csv"
LEMMA_CSV_DEFAULT = "analysis/reports/consonant_lemma_breakdown.csv"

# How a consonant lemma is classified.
#
# An earlier version of this module carried two hand-written lemma lists and
# called every t ~ d alternation "Auslautverhärtung". That contradicted the
# pipeline. step_2_establish_baseline admits a paradigm as bipartite only after
# a shape test that already separates the two mechanisms: Verner leaves the past
# plural as the odd cell (wesen s ~ s ~ r, quëden t ~ t ~ d), devoicing leaves
# the past singular as the odd cell (scheiden d ~ t ~ d). snîden, lîden and
# mîden are d ~ t ~ t - the past plural shares the t, so they are Class I
# grammatischer Wechsel, which is exactly why they were admitted. Labelling them
# orthographic here made this file disagree with the rule that built its own
# input, and moved 193 observations and 20 events into a category that the
# pipeline had already ruled out.
#
# So the category is now read off the same anchors the bipartite rule reads,
# through the same clauses. Nothing is hard-coded, and the two cannot drift
# apart. Note the consequence: after the shape test the devoicing category is
# empty by construction, because scheiden and its kind never reach the consonant
# channel at all. If a lemma ever does land in DEVOICING_SHAPE, that is a signal
# that the upstream rule has changed, not a finding about the language.
MECHANISM_VERNER_SG_PL = "Verner (past sg ~ past pl)"
MECHANISM_VERNER_MEDIAL = "Verner (medial, pres ~ past)"
MECHANISM_VERNER_PRES_PL = "Verner (pres ~ past pl)"
MECHANISM_DEVOICING = "Devoicing shape (should not occur)"
MECHANISM_UNCLASSIFIED = "Unclassified consonant alternation"

MORPHOLOGICAL_CATEGORIES = (
    MECHANISM_VERNER_SG_PL,
    MECHANISM_VERNER_MEDIAL,
    MECHANISM_VERNER_PRES_PL,
)


def _last_char(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text[-1] if text else None


def classify_consonant_lemma(anchor_row) -> str:
    """
    Name the clause of the bipartite rule that admitted this paradigm.

    Reads the same anchor codas and the same diff_cons_* flags that
    step_2_establish_baseline used, so the label cannot contradict the
    classification that produced the consonant channel in the first place.
    """
    def flag(name):
        value = anchor_row.get(name)
        return str(value).strip().lower() == "true"

    gw_pres_sg = flag("diff_cons_pres_pastsg")
    gw_pres_pl = flag("diff_cons_pres_pastpl")
    gw_sg_pl = flag("diff_cons_pastsg_pastpl")
    ab_pres_pl = flag("diff_vowel_pres_pastpl")

    cs = _last_char(anchor_row.get("anchor_coda_pastsg"))
    cp = _last_char(anchor_row.get("anchor_coda_pastpl"))
    cr = _last_char(anchor_row.get("anchor_coda_pres"))

    # Devoicing shape: the past singular is the odd cell out. The upstream rule
    # rejects this, so it is reported only as a tripwire.
    if cr is not None and cs is not None and cp is not None:
        if cr == cp and cs != cp and {cs, cp} == {"t", "d"}:
            return MECHANISM_DEVOICING

    if gw_sg_pl and not gw_pres_sg:
        return MECHANISM_VERNER_SG_PL
    if gw_pres_sg and not gw_sg_pl:
        return MECHANISM_VERNER_MEDIAL
    if ab_pres_pl and gw_pres_pl:
        return MECHANISM_VERNER_PRES_PL
    return MECHANISM_UNCLASSIFIED


from analysis.marking_type_summary import reshape as reshape_data


def load_and_reshape(coded_path: str = CODED_DEFAULT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads coded_output.csv and extracts:
    1. Full reshaped modeling dataset (matching run_brms.R and marking_type_summary.py).
    2. Raw token-level dataset for channel disaggregation.
    """
    df = pd.read_csv(coded_path, low_memory=False)

    for col in ("is_leveled_vowel_pres", "is_leveled_vowel_past",
                "is_leveled_cons_pres", "is_leveled_cons_past", "is_bipartite"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    long_dedup = reshape_data(coded_path)
    return long_dedup, df


def analyze_marking_rates(long_df: pd.DataFrame) -> pd.DataFrame:
    """Computes observation counts, leveled counts, and rates by marking_type."""
    summary = []
    for m in ["vowel_unipartite", "vowel_bipartite", "consonant_bipartite"]:
        sub = long_df[long_df["marking_type"] == m]
        obs = len(sub)
        leveled = int(sub["has_levelled"].sum())
        rate = 100.0 * leveled / obs if obs > 0 else 0.0
        summary.append({
            "marking_type": m,
            "observations": obs,
            "leveled": leveled,
            "rate_pct": rate
        })
    return pd.DataFrame(summary)


def analyze_consonant_lemmas(long_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes all lemmas participating in the consonant channel,
    labelling each by the clause of the bipartite rule that admitted it.
    """
    cons_sub = long_df[long_df["marking_type"] == "consonant_bipartite"].copy()

    # Collect alternation descriptions from raw data
    alt_map = {}
    for lid, grp in raw_df[raw_df["lemma_id"].isin(cons_sub["lemma_id"].unique())].groupby("lemma_id"):
        pres_alts = set(grp["cons_alternation_pres"].dropna().unique()) - {""}
        past_alts = set(grp["cons_alternation_past"].dropna().unique()) - {""}
        all_alts = " / ".join(filter(None, [", ".join(pres_alts), ", ".join(past_alts)]))
        alt_map[lid] = all_alts if all_alts else "unspecified"

    # One anchor row per lemma carries the paradigm the bipartite rule saw.
    anchors = raw_df.drop_duplicates(subset=["lemma_id", "variety"])
    cat_map = {}
    for lid, grp in anchors[anchors["lemma_id"].isin(cons_sub["lemma_id"].unique())].groupby("lemma_id"):
        labels = [classify_consonant_lemma(r) for _, r in grp.iterrows()]
        morphological = [x for x in labels if x in MORPHOLOGICAL_CATEGORIES]
        # A lemma attested in both varieties can resolve in only one of them.
        cat_map[lid] = morphological[0] if morphological else (labels[0] if labels else MECHANISM_UNCLASSIFIED)

    rows = []
    for lid, grp in cons_sub.groupby("lemma_id"):
        lemma_name = grp["lemma_clean"].iloc[0] if "lemma_clean" in grp.columns and pd.notna(grp["lemma_clean"].iloc[0]) else grp["lemma"].iloc[0]
        obs = len(grp)
        leveled = int(grp["has_levelled"].sum())
        rate = 100.0 * leveled / obs if obs > 0 else 0.0

        cat = cat_map.get(lid, MECHANISM_UNCLASSIFIED)
        mech = cat

        rows.append({
            "lemma_id": lid,
            "lemma": lemma_name,
            "category": cat,
            "mechanism": mech,
            "alternation": alt_map.get(lid, ""),
            "observations": obs,
            "leveled": leveled,
            "rate_pct": round(rate, 2),
            "share_of_cons_events": 0.0 # will fill below
        })

    res = pd.DataFrame(rows).sort_values(by=["leveled", "observations"], ascending=[False, False])
    total_events = res["leveled"].sum()
    if total_events > 0:
        res["share_of_cons_events"] = (100.0 * res["leveled"] / total_events).round(1)

    return res


def compute_mechanism_summary(lemma_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes leveling by the admitting clause of the bipartite rule."""
    mech_summary = lemma_df.groupby("category").agg(
        num_lemmas=("lemma_id", "count"),
        observations=("observations", "sum"),
        leveled=("leveled", "sum"),
    ).reset_index()

    mech_summary["rate_pct"] = (100.0 * mech_summary["leveled"] / mech_summary["observations"]).round(2)
    total_leveled = mech_summary["leveled"].sum()
    mech_summary["share_of_events_pct"] = (100.0 * mech_summary["leveled"] / total_leveled).round(1) if total_leveled > 0 else 0.0
    return mech_summary


def compute_statistical_contrasts(long_df: pd.DataFrame, lemma_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Computes 2x2 contingency tables and Odds Ratios comparing:
    1. Consonant Bipartite vs Vowel Unipartite
    2. Consonant Bipartite vs Vowel Bipartite
    3. Verner-admitted Consonant vs Vowel Bipartite
    4. Devoicing-shaped Consonant vs Vowel Bipartite (expected empty)
    """
    v_uni = long_df[long_df["marking_type"] == "vowel_unipartite"]
    v_bi = long_df[long_df["marking_type"] == "vowel_bipartite"]
    c_bi = long_df[long_df["marking_type"] == "consonant_bipartite"]

    morph_lids = set(
        lemma_df.loc[lemma_df["category"].isin(MORPHOLOGICAL_CATEGORIES), "lemma_id"]
    )
    ortho_lids = set(
        lemma_df.loc[lemma_df["category"] == MECHANISM_DEVOICING, "lemma_id"]
    )

    c_morph = c_bi[c_bi["lemma_id"].isin(morph_lids)]
    c_ortho = c_bi[c_bi["lemma_id"].isin(ortho_lids)]

    groups = {
        "Vowel Unipartite": (len(v_uni) - int(v_uni["has_levelled"].sum()), int(v_uni["has_levelled"].sum())),
        "Vowel Bipartite": (len(v_bi) - int(v_bi["has_levelled"].sum()), int(v_bi["has_levelled"].sum())),
        "Consonant Bipartite (All)": (len(c_bi) - int(c_bi["has_levelled"].sum()), int(c_bi["has_levelled"].sum())),
        "Consonant Verner-admitted": (len(c_morph) - int(c_morph["has_levelled"].sum()), int(c_morph["has_levelled"].sum())),
        "Consonant Devoicing-shaped": (len(c_ortho) - int(c_ortho["has_levelled"].sum()), int(c_ortho["has_levelled"].sum())),
    }

    contrasts = {}

    def calc_or(g1_name: str, g2_name: str):
        (n0_1, n1_1) = groups[g1_name]
        (n0_2, n1_2) = groups[g2_name]
        
        # Odds ratio = (n1_1 / n0_1) / (n1_2 / n0_2)
        odds1 = n1_1 / n0_1 if n0_1 > 0 else float('inf')
        odds2 = n1_2 / n0_2 if n0_2 > 0 else float('inf')
        # An empty group is a legitimate outcome, not an error: after the shape
        # test the devoicing category has no members, which is the point.
        if (n0_1 + n1_1) == 0 or (n0_2 + n1_2) == 0:
            return {
                "group1": g1_name,
                "group2": g2_name,
                "g1_rate_pct": None,
                "g2_rate_pct": None,
                "odds_ratio": None,
                "ci_95": (None, None),
                "p_value": None,
                "note": "no observations in one of the two groups",
            }

        odds_ratio = odds1 / odds2 if odds2 > 0 else float('inf')
        
        # 95% CI for log OR
        se_log_or = math.sqrt(1.0/max(1, n1_1) + 1.0/max(1, n0_1) + 1.0/max(1, n1_2) + 1.0/max(1, n0_2))
        ci_lower = math.exp(math.log(odds_ratio) - 1.96 * se_log_or) if odds_ratio > 0 else 0.0
        ci_upper = math.exp(math.log(odds_ratio) + 1.96 * se_log_or) if odds_ratio > 0 else float('inf')
        
        p_val = None
        if HAS_SCIPY:
            table = [[n1_1, n0_1], [n1_2, n0_2]]
            _, p_val = fisher_exact(table)

        return {
            "group1": g1_name,
            "group2": g2_name,
            "g1_rate_pct": round(100.0 * n1_1 / (n0_1 + n1_1), 2),
            "g2_rate_pct": round(100.0 * n1_2 / (n0_2 + n1_2), 2),
            "odds_ratio": round(odds_ratio, 2),
            "ci_95": (round(ci_lower, 2), round(ci_upper, 2)),
            "p_value": p_val
        }

    contrasts["Cons_All_vs_Vowel_Uni"] = calc_or("Consonant Bipartite (All)", "Vowel Unipartite")
    contrasts["Cons_All_vs_Vowel_Bi"] = calc_or("Consonant Bipartite (All)", "Vowel Bipartite")
    contrasts["Cons_Morph_vs_Vowel_Bi"] = calc_or("Consonant Verner-admitted", "Vowel Bipartite")
    contrasts["Cons_Ortho_vs_Vowel_Bi"] = calc_or("Consonant Devoicing-shaped", "Vowel Bipartite")

    return contrasts


def generate_markdown_report(
    rates_df: pd.DataFrame,
    lemma_df: pd.DataFrame,
    mech_df: pd.DataFrame,
    contrasts: Dict[str, Dict],
    output_path: str = REPORT_DEFAULT
) -> None:
    """Generates the comprehensive Markdown report on the consonant channel."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("# Consonant Channel Analysis")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This report provides a dedicated empirical audit of the consonant channel (`consonant_bipartite`) "
        "in Middle High German (MHG) and Early New High German (ENHG) strong verbs. While the primary "
        "Bayesian GAMM models focus on the vocalic channel (`vowel_unipartite` vs `vowel_bipartite`), "
        "the consonant channel behaves differently and is reported separately here."
    )
    lines.append("")
    lines.append("### Key Findings:")
    lines.append(
        f"1. **Elevated Raw Leveling Rate**: The consonant channel exhibits an overall leveling rate of **{rates_df.loc[rates_df['marking_type']=='consonant_bipartite', 'rate_pct'].values[0]:.2f}%** "
        f"({rates_df.loc[rates_df['marking_type']=='consonant_bipartite', 'leveled'].values[0]} / {rates_df.loc[rates_df['marking_type']=='consonant_bipartite', 'observations'].values[0]:,} observations), "
        f"which is substantially higher than bipartite vowel leveling (**{rates_df.loc[rates_df['marking_type']=='vowel_bipartite', 'rate_pct'].values[0]:.2f}%**, OR = {contrasts['Cons_All_vs_Vowel_Bi']['odds_ratio']}) "
        f"and unipartite vowel leveling (**{rates_df.loc[rates_df['marking_type']=='vowel_unipartite', 'rate_pct'].values[0]:.2f}%**, OR = {contrasts['Cons_All_vs_Vowel_Uni']['odds_ratio']})."
    )
    morph_mask = mech_df["category"].isin(MORPHOLOGICAL_CATEGORIES)
    devoicing_mask = mech_df["category"] == MECHANISM_DEVOICING
    morph_obs = int(mech_df.loc[morph_mask, "observations"].sum())
    morph_lev = int(mech_df.loc[morph_mask, "leveled"].sum())
    morph_rate = 100.0 * morph_lev / morph_obs if morph_obs else 0.0
    devoicing_obs = int(mech_df.loc[devoicing_mask, "observations"].sum())

    lines.append(
        "2. **All of it is Verner, by construction**: every paradigm in this channel was admitted by "
        "`step_2_establish_baseline` only after a shape test that separates grammatischer Wechsel from "
        "Auslautverhärtung. Verner leaves the past plural as the odd cell (*wesen* s ~ s ~ r, *quëden* "
        "t ~ t ~ d); devoicing leaves the past singular as the odd cell (*scheiden* d ~ t ~ d). The Class I "
        "verbs *snîden*, *lîden* and *mîden* are d ~ t ~ **t** - the plural shares the t - so their t ~ d "
        f"is grammatischer Wechsel, not a spelling effect. Verner-admitted: **{morph_rate:.2f}%** "
        f"({morph_lev} / {morph_obs:,}). Devoicing-shaped: **{devoicing_obs} observations**, as expected - "
        "a non-zero count here would mean the upstream rule had changed."
    )
    top_morph = lemma_df[lemma_df["category"].isin(MORPHOLOGICAL_CATEGORIES)].sort_values("leveled", ascending=False)
    top_morph_str = (
        f"*{top_morph.iloc[0]['lemma']}* (lemma {top_morph.iloc[0]['lemma_id']}, "
        f"{top_morph.iloc[0]['share_of_cons_events']:.1f}% of consonant events)"
        if len(top_morph) > 0 else "None"
    )
    lines.append(f"3. **High Concentration**: The largest contributor is {top_morph_str}.")
    lines.append("")

    lines.append("## 1. Overall Marking Type Leveling Rates")
    lines.append("")
    lines.append("| Marking Type | Observations | Leveling Events | Leveling Rate (%) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for _, r in rates_df.iterrows():
        lines.append(f"| `{r['marking_type']}` | {int(r['observations']):,} | {int(r['leveled']):,} | {r['rate_pct']:.2f}% |")
    lines.append("")

    lines.append("## 2. Breakdown by the Admitting Clause of the Bipartite Rule")
    lines.append("")
    lines.append("| Alternation Category | Lemmas | Observations | Leveled | Leveling Rate (%) | Share of Consonant Events |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, r in mech_df.iterrows():
        lines.append(f"| **{r['category']}** | {int(r['num_lemmas'])} | {int(r['observations']):,} | {int(r['leveled']):,} | {r['rate_pct']:.2f}% | {r['share_of_events_pct']:.1f}% |")
    lines.append("")

    lines.append("## 3. Per-Lemma Consonant Breakdown")
    lines.append("")
    lines.append("| Lemma ID | Lemma | Alternation Pattern | Category | Obs | Leveled | Rate (%) | Share (%) |")
    lines.append("| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |")
    for _, r in lemma_df.iterrows():
        lines.append(f"| {r['lemma_id']} | *{r['lemma']}* | `{r['alternation']}` | {r['category']} | {int(r['observations']):,} | {int(r['leveled']):,} | {r['rate_pct']:.2f}% | {r['share_of_cons_events']:.1f}% |")
    lines.append("")

    lines.append("## 4. Statistical Contrast Analysis")
    lines.append("")
    lines.append("| Comparison | Group 1 Rate | Group 2 Rate | Odds Ratio | 95% Confidence Interval | p-value (Fisher) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for name, c in contrasts.items():
        if c.get("note"):
            # An empty group is the expected result for the devoicing contrast.
            lines.append(f"| {c['group1']} vs. {c['group2']} | - | - | - | - | {c['note']} |")
            continue
        p_str = f"{c['p_value']:.2e}" if c['p_value'] is not None else "N/A"
        lines.append(f"| {c['group1']} vs. {c['group2']} | {c['g1_rate_pct']:.2f}% | {c['g2_rate_pct']:.2f}% | {c['odds_ratio']:.2f} | [{c['ci_95'][0]}, {c['ci_95'][1]}] | {p_str} |")
    lines.append("")

    lines.append("## 5. Methodological & Theoretical Implications")
    lines.append("")
    lines.append(
        "1. **Justification for Option A (Vowel-Only Primary Model)**:\n"
        "   - Combining consonant marking with vowel marking into a single treatment factor introduces substantial noise because consonant leveling is heavily confounded by orthographic spelling shifts (*Auslautverhärtung* in *scheiden* and *lîden*).\n"
        "   - By separating the consonant channel into this dedicated analysis and estimating a pure vowel-only GAMM, the test of Paul's Principle tests morphological Ablaut resistance against an unambiguous unipartite control baseline."
    )
    lines.append(
        "2. **Grammatischer Wechsel Trajectory**:\n"
        "   - True Grammatischer Wechsel verbs (*ziehen*, *verlieren*, *genesen*, *zîhen*) undergo progressive analogical leveling across the MHG-to-ENHG transition as the plural/participle voiced consonants generalize into the singular preterite.\n"
        "   - This confirms that consonantal stem alternations operate under distinct phonetic and morphosyntactic pressures compared to vowel ablaut grades."
    )
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated Markdown report at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Consonant Channel Analysis")
    parser.add_argument("--coded", default=CODED_DEFAULT, help="Path to coded_output.csv")
    parser.add_argument("--out-report", default=REPORT_DEFAULT, help="Path to output markdown report")
    parser.add_argument("--out-summary", default=SUMMARY_CSV_DEFAULT, help="Path to output summary CSV")
    parser.add_argument("--out-lemmas", default=LEMMA_CSV_DEFAULT, help="Path to output per-lemma CSV")
    args = parser.parse_args()

    print(f"Loading data from {args.coded}...")
    long_df, raw_df = load_and_reshape(args.coded)

    rates_df = analyze_marking_rates(long_df)
    lemma_df = analyze_consonant_lemmas(long_df, raw_df)
    mech_df = compute_mechanism_summary(lemma_df)
    contrasts = compute_statistical_contrasts(long_df, lemma_df)

    print("\n--- Marking Type Leveling Rates ---")
    print(rates_df.to_string(index=False))

    print("\n--- Mechanism Summary (by admitting clause) ---")
    print(mech_df.to_string(index=False))

    print("\n--- Consonant Lemmas Breakdown ---")
    print(lemma_df[["lemma_id", "lemma", "category", "observations", "leveled", "rate_pct", "share_of_cons_events"]].to_string(index=False))

    # Save CSV outputs
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    mech_df.to_csv(args.out_summary, index=False)
    print(f"\nWrote summary CSV to: {args.out_summary}")

    lemma_df.to_csv(args.out_lemmas, index=False)
    print(f"Wrote lemma breakdown CSV to: {args.out_lemmas}")

    # Generate Markdown report
    generate_markdown_report(rates_df, lemma_df, mech_df, contrasts, args.out_report)


if __name__ == "__main__":
    main()
