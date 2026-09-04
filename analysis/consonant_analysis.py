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
   (~7.35%) substantially higher than bipartite vowel leveling (~0.80%) and
   unipartite vowel leveling (~2.04%)? These figures move whenever the coding
   changes; the report recomputes them, so treat the numbers here as indicative.
2. Which clause of the bipartite rule admitted each paradigm, read off the same
   anchors that step_2_establish_baseline used. Devoicing-shaped paradigms
   (scheiden d ~ t ~ d) are already excluded upstream, so that category is
   expected to be empty here; it is reported as a tripwire on the rule, not as
   a finding about the language.
3. Concentration across Lemmas: Which verbs drive the consonant leveling counts?
4. Channel Asymmetry: within a bipartite cell the vowel row and the consonant row
   describe the same text, so they are a matched pair. When exactly one of the two
   marks gives way, which one is it? This is tested on the discordant pairs, with
   an interval bootstrapped over lemmas; the unpaired contrasts in section 4 of
   the report treat those matched rows as independent and are descriptive only.

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
    from scipy.stats import fisher_exact, chi2_contingency, binomtest
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


CODED_DEFAULT = "data/coded_output.csv"
REPORT_DEFAULT = "analysis/reports/consonant_analysis_report.md"
SUMMARY_CSV_DEFAULT = "analysis/reports/consonant_summary.csv"
LEMMA_CSV_DEFAULT = "analysis/reports/consonant_lemma_breakdown.csv"
PAIRED_CSV_DEFAULT = "analysis/reports/consonant_paired_discordance.csv"

# The key that identifies one observation cell. run_brms.R de-duplicates on the
# whole predictor row, and `id` is a document id rather than a token id, so a
# cell is one lemma in one document in one inflectional slot - not one token.
# The vowel and the consonant row of the same cell are the same stretch of text
# written by the same scribe, which is what makes them a matched pair.
PAIR_KEY = ["lemma_id", "id", "date", "variety", "std_infl", "corpus"]
PAIRED_BOOTSTRAP_DRAWS = 10000
PAIRED_BOOTSTRAP_SEED = 97

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


def build_paired_cells(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match each bipartite cell's vowel row to its consonant row.

    Only bipartite paradigms have a consonant row at all, so this is the whole
    population in which the two channels can be compared. Pairing them is not a
    refinement of the marking_type contrast - it is a different design. The
    bipartite-vs-unipartite contrast is between lemmas; vowel-vs-consonant is
    within one cell, where date, scribe, document, frequency and inflectional
    slot are identical by construction and cancel.
    """
    vowel = (
        long_df[long_df["marking_type"] == "vowel_bipartite"]
        .drop_duplicates(subset=PAIR_KEY)[PAIR_KEY + ["lemma", "has_levelled"]]
        .rename(columns={"has_levelled": "vowel_leveled"})
    )
    cons = (
        long_df[long_df["marking_type"] == "consonant_bipartite"]
        .drop_duplicates(subset=PAIR_KEY)[PAIR_KEY + ["has_levelled"]]
        .rename(columns={"has_levelled": "cons_leveled"})
    )
    return vowel.merge(cons, on=PAIR_KEY, how="inner")


def paired_channel_test(pairs: pd.DataFrame) -> Dict:
    """
    Which channel gives way first, tested within the cell.

    Concordant pairs carry no information about the direction of the asymmetry,
    so the test is the exact binomial on the discordant ones - McNemar's test in
    its exact form.

    The interval is bootstrapped over lemmas, not over pairs. Events are heavily
    concentrated (ziehen alone supplies most of them), and an interval that
    resamples pairs would treat one verb's many cells as many independent facts.
    Resampling lemmas asks the question that matters: would this hold on another
    sample of verbs?
    """
    n = len(pairs)
    both = int(((pairs["vowel_leveled"] == 1) & (pairs["cons_leveled"] == 1)).sum())
    neither = int(((pairs["vowel_leveled"] == 0) & (pairs["cons_leveled"] == 0)).sum())
    cons_only = int(((pairs["vowel_leveled"] == 0) & (pairs["cons_leveled"] == 1)).sum())
    vowel_only = int(((pairs["vowel_leveled"] == 1) & (pairs["cons_leveled"] == 0)).sum())
    discordant = cons_only + vowel_only

    point = cons_only / discordant if discordant else float("nan")

    p_value = None
    if HAS_SCIPY and discordant:
        p_value = binomtest(cons_only, discordant, 0.5).pvalue

    # Lemma-clustered bootstrap: resample whole verbs with replacement.
    rng = np.random.default_rng(PAIRED_BOOTSTRAP_SEED)
    lemmas = pairs["lemma_id"].unique()
    by_lemma = {
        lid: (
            int(((g["vowel_leveled"] == 0) & (g["cons_leveled"] == 1)).sum()),
            int(((g["vowel_leveled"] == 1) & (g["cons_leveled"] == 0)).sum()),
        )
        for lid, g in pairs.groupby("lemma_id")
    }
    draws = []
    for _ in range(PAIRED_BOOTSTRAP_DRAWS):
        picked = rng.choice(lemmas, size=len(lemmas), replace=True)
        c = sum(by_lemma[l][0] for l in picked)
        v = sum(by_lemma[l][1] for l in picked)
        if c + v:
            draws.append(c / (c + v))
    draws = np.array(draws)

    return {
        "n_pairs": n,
        "n_lemmas": int(pairs["lemma_id"].nunique()),
        "both": both,
        "neither": neither,
        "cons_only": cons_only,
        "vowel_only": vowel_only,
        "discordant": discordant,
        "point_pct": 100.0 * point,
        "p_value": p_value,
        "ci_lower_pct": 100.0 * float(np.quantile(draws, 0.025)) if len(draws) else None,
        "ci_upper_pct": 100.0 * float(np.quantile(draws, 0.975)) if len(draws) else None,
        "share_reversed_pct": 100.0 * float((draws < 0.5).mean()) if len(draws) else None,
        "draws_used": int(len(draws)),
    }


def paired_lemma_breakdown(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-verb discordance, so the reader can see how concentrated the result is."""
    rows = []
    for lid, g in pairs.groupby("lemma_id"):
        cons_only = int(((g["vowel_leveled"] == 0) & (g["cons_leveled"] == 1)).sum())
        vowel_only = int(((g["vowel_leveled"] == 1) & (g["cons_leveled"] == 0)).sum())
        rows.append({
            "lemma_id": lid,
            "lemma": g["lemma"].iloc[0],
            "paired_cells": len(g),
            "cons_only": cons_only,
            "vowel_only": vowel_only,
            "discordant": cons_only + vowel_only,
        })
    out = pd.DataFrame(rows).sort_values("discordant", ascending=False)
    total = out["discordant"].sum()
    out["share_of_discordant_pct"] = (
        (100.0 * out["discordant"] / total).round(1) if total else 0.0
    )
    return out


def generate_markdown_report(
    rates_df: pd.DataFrame,
    lemma_df: pd.DataFrame,
    mech_df: pd.DataFrame,
    contrasts: Dict[str, Dict],
    paired: Dict,
    paired_df: pd.DataFrame,
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
    lines.append(
        f"4. **Within the cell, the consonant gives way first**: on the {paired['n_pairs']:,} cells where both "
        f"channels are informative, exactly one mark gives way in {paired['discordant']} of them, and it is the "
        f"consonant in **{paired['point_pct']:.1f}%** of those "
        f"(lemma-clustered 95% CI {paired['ci_lower_pct']:.1f}%-{paired['ci_upper_pct']:.1f}%). "
        "This is the comparison the channel question actually asks, and it is reported in section 5."
    )
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

    lines.append("## 4. Statistical Contrast Analysis (Unpaired - Descriptive Only)")
    lines.append("")
    lines.append(
        "> **Read these as descriptive rates, not as tests.** The consonant rows and the vowel-bipartite rows "
        f"are not independent samples: {paired['n_pairs']:,} of them are the *same cells*, each contributing one "
        "row to each channel. Fisher's exact test assumes independence, so the p-values below are far smaller "
        "than the evidence warrants, and the events are concentrated in a handful of verbs besides. "
        "The consonant-vs-vowel comparison is tested properly in section 5, which uses the pairing instead of "
        "ignoring it. The `Vowel Unipartite` row is a between-lemma contrast and is reported for scale only; "
        "the modelled version of that contrast is the GAMM in `analysis/run_brms.R`."
    )
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

    lines.append("## 5. Within-Cell Channel Asymmetry (Paired Design)")
    lines.append("")
    lines.append(
        "Sections 1-4 compare channels as if they were separate samples. They are not. Every bipartite cell "
        "carries a vowel row and a consonant row describing the same stretch of text, so the two are a matched "
        "pair: same verb, same document, same date, same scribe, same inflectional slot, same frequency. "
        "Everything the GAMM spends its covariates controlling for cancels by construction here. "
        "The question this design answers is not *how much* each channel levels, but **which mark gives way "
        "when only one of them does**."
    )
    lines.append("")
    lines.append(f"Matched cells: **{paired['n_pairs']:,}** across **{paired['n_lemmas']}** bipartite verbs.")
    lines.append("")
    lines.append("| | Consonant resisted | Consonant leveled |")
    lines.append("| :--- | :---: | :---: |")
    lines.append(f"| **Vowel resisted** | {paired['neither']:,} | {paired['cons_only']} |")
    lines.append(f"| **Vowel leveled** | {paired['vowel_only']} | {paired['both']} |")
    lines.append("")
    lines.append(
        "Concordant cells (both marks resisted, or both gave way) carry no information about direction, so the "
        f"test is the exact binomial on the **{paired['discordant']} discordant** cells - McNemar's test in its "
        "exact form."
    )
    lines.append("")
    p_str = f"{paired['p_value']:.2e}" if paired.get("p_value") is not None else "N/A (scipy unavailable)"
    lines.append("| Quantity | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Discordant cells | {paired['discordant']} |")
    lines.append(f"| Consonant gave way | {paired['cons_only']} |")
    lines.append(f"| Vowel gave way | {paired['vowel_only']} |")
    lines.append(f"| P(the mark that gives way is the consonant) | **{paired['point_pct']:.1f}%** |")
    lines.append(f"| Exact binomial p (vs 50%) | {p_str} |")
    lines.append(
        f"| Lemma-clustered 95% CI | ({paired['ci_lower_pct']:.1f}%, {paired['ci_upper_pct']:.1f}%) |"
    )
    lines.append(f"| Bootstrap draws reversing the direction | {paired['share_reversed_pct']:.1f}% |")
    lines.append("")
    lines.append(
        "The interval resamples **verbs**, not cells. The events are concentrated, and an interval built by "
        "resampling cells would count one verb's many documents as many independent facts. The clustered "
        "interval is therefore much wider than the exact p-value suggests, and it is the one to quote."
    )
    lines.append("")
    lines.append("### 5.1 Where the discordant cells come from")
    lines.append("")
    lines.append("| Lemma ID | Lemma | Paired Cells | Consonant Only | Vowel Only | Discordant | Share (%) |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, r in paired_df.iterrows():
        lines.append(
            f"| {r['lemma_id']} | *{r['lemma']}* | {int(r['paired_cells']):,} | {int(r['cons_only'])} | "
            f"{int(r['vowel_only'])} | {int(r['discordant'])} | {r['share_of_discordant_pct']:.1f}% |"
        )
    lines.append("")
    lines.append("### 5.2 What this does and does not support")
    lines.append("")
    lines.append(
        "1. **It refines Paul rather than contradicting him.** Paul's argument is that two marks reinforce each "
        "other. If one of them erodes several times faster than the other, the bipartite state is transient and "
        "asymmetric: the grammatischer Wechsel is the weak link, and bipartite marking is a way-station rather "
        "than a stable configuration."
    )
    lines.append(
        "2. **It is a separate result from the GAMM, with a separate design.** The bipartite-vs-unipartite "
        "contrast is between lemmas and rests on few verbs. This one is within the cell. Neither is a robustness "
        "check on the other, and they should be reported as two findings, not one."
    )
    lines.append(
        "3. **It is concentrated.** Read section 5.1 before quoting the percentage. The clustered interval "
        "already reflects that concentration; the point estimate does not."
    )
    lines.append(
        "4. **It does not license adding the consonant channel to `marking_type` as a third level.** Unipartite "
        "verbs have no consonant rows by construction, so that level would have no comparison group; the paired "
        "rows would enter the GAMM as if independent; and one random-effect structure cannot serve a "
        "between-lemma and a within-cell contrast at once. That is why `run_brms.R` fits the vowel channel only."
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
    parser.add_argument("--out-paired", default=PAIRED_CSV_DEFAULT,
                        help="Path to output per-lemma paired discordance CSV")
    args = parser.parse_args()

    print(f"Loading data from {args.coded}...")
    long_df, raw_df = load_and_reshape(args.coded)

    rates_df = analyze_marking_rates(long_df)
    lemma_df = analyze_consonant_lemmas(long_df, raw_df)
    mech_df = compute_mechanism_summary(lemma_df)
    contrasts = compute_statistical_contrasts(long_df, lemma_df)
    pairs = build_paired_cells(long_df)
    paired = paired_channel_test(pairs)
    paired_df = paired_lemma_breakdown(pairs)

    print("\n--- Marking Type Leveling Rates ---")
    print(rates_df.to_string(index=False))

    print("\n--- Mechanism Summary (by admitting clause) ---")
    print(mech_df.to_string(index=False))

    print("\n--- Within-Cell Channel Asymmetry (paired) ---")
    print(f"  matched cells      {paired['n_pairs']:,} across {paired['n_lemmas']} verbs")
    print(f"  discordant         {paired['discordant']}  (consonant {paired['cons_only']}, vowel {paired['vowel_only']})")
    print(f"  P(consonant first) {paired['point_pct']:.1f}%  "
          f"lemma-clustered 95% CI ({paired['ci_lower_pct']:.1f}%, {paired['ci_upper_pct']:.1f}%)")
    if paired["p_value"] is not None:
        print(f"  exact binomial p   {paired['p_value']:.3e}")
    print(paired_df.to_string(index=False))

    print("\n--- Consonant Lemmas Breakdown ---")
    print(lemma_df[["lemma_id", "lemma", "category", "observations", "leveled", "rate_pct", "share_of_cons_events"]].to_string(index=False))

    # Save CSV outputs
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    mech_df.to_csv(args.out_summary, index=False)
    print(f"\nWrote summary CSV to: {args.out_summary}")

    lemma_df.to_csv(args.out_lemmas, index=False)
    print(f"Wrote lemma breakdown CSV to: {args.out_lemmas}")

    paired_df.to_csv(args.out_paired, index=False)
    print(f"Wrote paired discordance CSV to: {args.out_paired}")

    # Generate Markdown report
    generate_markdown_report(rates_df, lemma_df, mech_df, contrasts, paired, paired_df, args.out_report)


if __name__ == "__main__":
    main()
