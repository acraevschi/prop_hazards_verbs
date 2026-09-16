#!/usr/bin/env python3
"""
Marking-type summary, straight from data/coded_output.csv.

Why this exists
---------------
The marking_type and explicit model-observation counts should be available
before a Stan run begins. This script reproduces the reshape, audits the chosen
deduplication key, and writes the tables used to approve a production run.

This script does the reshape and nothing else. It is a faithful port of the
"2. Reshaping & Predictor Construction" block of run_brms.R (lines ~270-345):
same vowel_leveled_any / cons_leveled_any collapse, same two filters, same long
pivot, same marking_type construction, and the same token-identity validation.
It writes reports only, so it cannot disturb the fits.

If run_brms.R's reshape changes, this port has to change with it. The row count
it reports should equal the "Prepared N modeling observations" line that
run_brms.R prints.

What it prints
--------------
1. The source-token marking_type table: observations, leveling events, rate.
2. The explicit GAMM table after
   `document_id × lemma_id × std_infl × has_levelled` deduplication.
3. Bipartite vowel leveling by period, which is where the S-curve claim lives.
4. Per-lemma contribution to the bipartite vowel events. Read this one. The
   bipartite cell is small enough that a single lemma can carry it, and a rate
   that rests on one verb is a different claim from a rate that rests on twenty.

Sensitivity
-----------
--sensitivity re-runs the whole coding pipeline against an alternative target
table in which every preterite chosen by VARIANT_POLICY (data/extract_nhg_
preterites.py) is flipped to the variant that policy rejected - hieb->haute,
sott->siedete, and so on - and prints both tables side by side. That is the
honest way to show what the curator's variant calls are worth, since UniMorph
itself offers no way to rank them. It takes a few minutes and writes only to a
temporary directory.

Usage
-----
    python3 analysis/marking_type_summary.py
    python3 analysis/marking_type_summary.py --sensitivity
    python3 analysis/marking_type_summary.py --coded path/to/coded_output.csv
    python3 analysis/marking_type_summary.py --out analysis/reports/marking_type_summary.csv
"""

import argparse
import csv
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CODED = "data/coded_output.csv"
TARGETS = "data/lemmas/nhg_targets.csv"
REPORT_DIR = "analysis/reports"
SUMMARY_OUT = f"{REPORT_DIR}/marking_type_summary.csv"
REPORT_OUT = f"{REPORT_DIR}/marking_type_report.md"

MARKING_ORDER = ["vowel_unipartite", "vowel_bipartite", "consonant_bipartite"]
PERIODS = [(1050, 1200), (1200, 1350), (1350, 1500), (1500, 1650)]
MODEL_KEY = ["document_id", "lemma_id", "std_infl"]


def _leveled_any(a, b):
    """
    run_brms.R's case_when: 1 if either contrast leveled, 0 if either is an
    observed non-event, NA if both are missing. Order matters - the 1 test comes
    first, so a row that is 1 on one contrast and 0 on the other counts as 1.
    """
    out = pd.Series(pd.NA, index=a.index, dtype="Float64")
    out[(a == 0) | (b == 0)] = 0
    out[(a == 1) | (b == 1)] = 1
    return out


def _type_freq(df, alt_col):
    """calc_type_freq: distinct lemma_id per (variety, alternation pattern)."""
    sub = df[df[alt_col].notna() & (df[alt_col] != "")]
    freq = sub.groupby(["variety", alt_col])["lemma_id"].nunique().rename(f"{alt_col}_freq")
    return freq.reset_index()


def _validate_token_identity(df, context):
    """Fail rather than silently collapsing two records for one corpus token."""
    required = {"document_id", "token_id", "observation_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{context} is missing identity columns: {', '.join(sorted(missing))}"
        )
    if df[list(required)].isna().any().any():
        raise ValueError(f"{context} contains missing document or token identities")
    duplicates = df[df["observation_id"].duplicated(keep=False)]
    if not duplicates.empty:
        examples = duplicates["observation_id"].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{context} contains repeated observation_id values: {examples}"
        )


def reshape(coded_path=CODED):
    """Port of run_brms.R's base_model_data -> model_data. Returns the long frame."""
    df = pd.read_csv(coded_path, low_memory=False)
    _validate_token_identity(df, coded_path)

    for col in ("is_leveled_vowel_pres", "is_leveled_vowel_past",
                "is_leveled_cons_pres", "is_leveled_cons_past"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # run_brms.R replaces `lemma` with the shortest surface string per lemma_id
    # before anything else, so every downstream group sees that label rather
    # than the raw surface form.
    rep = (
        df[df["lemma"].notna() & (df["lemma"] != "")]
        .assign(_n=lambda d: d["lemma"].str.len())
        .sort_values(["lemma_id", "_n", "lemma"], kind="mergesort")
        .groupby("lemma_id")["lemma"]
        .first()
    )
    df["lemma"] = df["lemma_id"].map(rep)

    df["token_freq_avg"] = df.groupby(["lemma_id", "std_infl"])["form_freq_per_1000"].transform("mean")
    for alt_col in ("vowel_alternation_pres", "vowel_alternation_past",
                    "cons_alternation_pres", "cons_alternation_past"):
        df = df.merge(_type_freq(df, alt_col), on=["variety", alt_col], how="left")

    df["vowel_leveled_any"] = _leveled_any(df["is_leveled_vowel_pres"], df["is_leveled_vowel_past"])
    df["cons_leveled_any"] = _leveled_any(df["is_leveled_cons_pres"], df["is_leveled_cons_past"])

    df = df[df["is_bipartite"].notna()]
    df = df[df["vowel_leveled_any"].notna() | df["cons_leveled_any"].notna()]

    long = df.melt(
        id_vars=[c for c in df.columns if c not in ("vowel_leveled_any", "cons_leveled_any")],
        value_vars=["vowel_leveled_any", "cons_leveled_any"],
        var_name="element_type_raw",
        value_name="has_levelled",
    )
    long = long[long["has_levelled"].notna()].copy()
    long["element_type"] = long["element_type_raw"].map(
        {"vowel_leveled_any": "vowel", "cons_leveled_any": "consonant"}
    )

    bipartite = pd.to_numeric(long["is_bipartite"], errors="coerce")
    long["marking_type"] = pd.NA
    long.loc[(long["element_type"] == "vowel") & (bipartite == 0), "marking_type"] = "vowel_unipartite"
    long.loc[(long["element_type"] == "vowel") & (bipartite == 1), "marking_type"] = "vowel_bipartite"
    long.loc[(long["element_type"] == "consonant") & (bipartite == 1), "marking_type"] = "consonant_bipartite"

    # Predictor construction is token-level. observation_id remains the stable
    # unit while document_id is reserved for the document random effect.
    vowel = long["element_type"] == "vowel"
    long["target_alt_pres_freq"] = long["cons_alternation_pres_freq"].where(
        ~vowel, long["vowel_alternation_pres_freq"])
    long["target_alt_past_freq"] = long["cons_alternation_past_freq"].where(
        ~vowel, long["vowel_alternation_past_freq"])
    long["has_alt_pres"] = long["target_alt_pres_freq"].notna().map({True: "yes", False: "no"})
    long["has_alt_past"] = long["target_alt_past_freq"].notna().map({True: "yes", False: "no"})
    long["log_freq"] = np.log(pd.to_numeric(long["lemma_freq_per_1000"], errors="coerce") + 0.0001)
    long["log_token_freq"] = np.log(long["token_freq_avg"] + 0.0001)
    long["log_alt_pres_freq"] = np.where(
        long["has_alt_pres"] == "yes", np.log(long["target_alt_pres_freq"]), 0.0)
    long["log_alt_past_freq"] = np.where(
        long["has_alt_past"] == "yes", np.log(long["target_alt_past_freq"]), 0.0)

    keep = [
        "lemma", "lemma_id", "date", "log_freq", "log_token_freq",
        "has_alt_pres", "log_alt_pres_freq", "has_alt_past", "log_alt_past_freq",
        "marking_type", "is_bipartite", "element_type", "has_levelled",
        "document_id", "token_id", "observation_id",
        "variety", "std_infl", "corpus",
    ]
    long = long[keep]
    repeated_channels = long.duplicated(["observation_id", "marking_type"], keep=False)
    if repeated_channels.any():
        examples = long.loc[repeated_channels, "observation_id"].unique()[:5]
        raise ValueError(
            "One token produced multiple rows for the same marking channel: "
            + ", ".join(examples)
        )
    long["has_levelled"] = long["has_levelled"].astype(int)
    long["date"] = pd.to_numeric(long["date"], errors="coerce")
    return long


def marking_table(long):
    rows = []
    for marking in MARKING_ORDER:
        sub = long[long["marking_type"] == marking]
        total, lev = len(sub), int(sub["has_levelled"].sum())
        rows.append((marking, total, lev, 100 * lev / total if total else 0.0))
    sub = long[long["marking_type"].notna()]
    total, lev = len(sub), int(sub["has_levelled"].sum())
    rows.append(("total", total, lev, 100 * lev / total if total else 0.0))
    return rows


def marking_frame(long):
    return pd.DataFrame(
        marking_table(long),
        columns=["marking_type", "observations", "leveled", "leveling_pct"],
    )


def model_observation_frame(long):
    """Apply the explicit GAMM deduplication key to vowel outcomes.

    Repeated tokens with the same outcome inside a document-lemma-slot cell do
    not add likelihood weight. A mixed cell contributes one preserved and one
    leveled Bernoulli row, irrespective of how many tokens realize each state.
    Token counts are retained only for audit.
    """
    vowel = long[long["marking_type"].isin(
        ["vowel_unipartite", "vowel_bipartite"]
    )].copy()
    varying_marking = vowel.groupby(MODEL_KEY)["marking_type"].nunique()
    if (varying_marking > 1).any():
        raise ValueError("A document-lemma-slot cell has more than one marking type")

    counts = (
        vowel.groupby(MODEL_KEY)
        .agg(
            source_tokens=("has_levelled", "size"),
            leveled_tokens=("has_levelled", "sum"),
            marking_type=("marking_type", "first"),
            lemma=("lemma", "first"),
        )
        .reset_index()
    )
    counts["preserved_tokens"] = counts["source_tokens"] - counts["leveled_tokens"]
    rows = (
        vowel.drop_duplicates(MODEL_KEY + ["has_levelled"])
        [MODEL_KEY + ["has_levelled"]]
        .merge(counts, on=MODEL_KEY, how="left", validate="many_to_one")
    )
    rows["model_row_id"] = (
        rows["document_id"].astype(str) + "|"
        + rows["lemma_id"].astype(str) + "|"
        + rows["std_infl"].astype(str) + "|"
        + rows["has_levelled"].astype(str)
    )
    if rows["model_row_id"].duplicated().any():
        raise ValueError("Explicit document-lemma-slot-outcome key is not unique")
    return rows


def model_observation_summary(long):
    rows = model_observation_frame(long)
    summary = (
        rows.groupby("marking_type")["has_levelled"]
        .agg(observations="count", leveled="sum")
        .reindex(["vowel_unipartite", "vowel_bipartite"])
        .reset_index()
    )
    summary["preserved"] = summary["observations"] - summary["leveled"]
    summary["leveling_pct"] = 100 * summary["leveled"] / summary["observations"]
    total = pd.DataFrame(
        [{
            "marking_type": "total",
            "observations": int(summary["observations"].sum()),
            "leveled": int(summary["leveled"].sum()),
            "preserved": int(summary["preserved"].sum()),
            "leveling_pct": 100 * summary["leveled"].sum() / summary["observations"].sum(),
        }]
    )
    return pd.concat([summary, total], ignore_index=True)


def model_event_concentration(long):
    rows = model_observation_frame(long)
    bipartite = rows[rows["marking_type"] == "vowel_bipartite"]
    out = (
        bipartite.groupby(["lemma_id", "lemma"])["has_levelled"]
        .agg(observations="count", leveled="sum")
        .reset_index()
        .sort_values(["leveled", "observations"], ascending=False)
    )
    total = out["leveled"].sum()
    out["share_of_events_pct"] = 100 * out["leveled"] / total if total else 0.0
    return out


def period_frame(long):
    sub = long[(long["marking_type"] == "vowel_bipartite") & long["date"].notna()]
    rows = []
    for lo, hi in PERIODS:
        window = sub[(sub["date"] >= lo) & (sub["date"] < hi)]
        rows.append(
            {
                "period": f"[{lo}, {hi})",
                "observations": len(window),
                "leveled": int(window["has_levelled"].sum()),
                "leveling_pct": 100 * window["has_levelled"].mean() if len(window) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def concentration_frame(long):
    sub = long[long["marking_type"] == "vowel_bipartite"]
    out = (
        sub.groupby(["lemma_id", "lemma"])["has_levelled"]
        .agg(observations="count", leveled="sum")
        .reset_index()
        .sort_values(["leveled", "observations"], ascending=False)
    )
    total = out["leveled"].sum()
    out["share_of_events_pct"] = 100 * out["leveled"] / total if total else 0.0
    return out


def channel_frame(coded_path):
    df = pd.read_csv(coded_path, low_memory=False)
    _validate_token_identity(df, coded_path)
    for col in ("is_leveled_vowel_pres", "is_leveled_vowel_past",
                "is_leveled_cons_pres", "is_leveled_cons_past", "is_bipartite"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["is_bipartite"].notna()]
    rows = []
    channels = [
        ("is_leveled_vowel_past", "vowel (past sg ~ past pl)", (0, 1)),
        ("is_leveled_vowel_pres", "vowel (pres ~ past)", (0, 1)),
        ("is_leveled_cons_past", "cons (past sg ~ past pl)", (1,)),
        ("is_leveled_cons_pres", "cons (pres ~ past)", (1,)),
    ]
    names = {0: "unipartite", 1: "bipartite"}
    for column, channel, markings in channels:
        present = df[df[column].notna()]
        for marking in markings:
            obs = present[present["is_bipartite"] == marking]
            rows.append(
                {
                    "channel": channel,
                    "marking": names[marking],
                    "observations": len(obs),
                    "leveled": int(obs[column].sum()),
                    "leveling_pct": 100 * obs[column].mean() if len(obs) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _scenario_row(name, long):
    model_rows = model_observation_frame(long)
    bi = model_rows[model_rows["marking_type"] == "vowel_bipartite"]
    uni = model_rows[model_rows["marking_type"] == "vowel_unipartite"]
    bi_rate = bi["has_levelled"].mean() if len(bi) else np.nan
    uni_rate = uni["has_levelled"].mean() if len(uni) else np.nan
    return {
        "scenario": name,
        "bipartite_observations": len(bi),
        "bipartite_events": int(bi["has_levelled"].sum()),
        "bipartite_rate_pct": 100 * bi_rate,
        "unipartite_observations": len(uni),
        "unipartite_events": int(uni["has_levelled"].sum()),
        "unipartite_rate_pct": 100 * uni_rate,
        "unipartite_to_bipartite_rate_ratio": uni_rate / bi_rate if bi_rate else np.nan,
    }


def run_lihen_sensitivity(coded_path, workdir):
    """Recompute the three documented treatments of the weak lîhen anchor."""
    from data.corpus_approach_coding import (
        step_1_preprocessing,
        step_2_establish_baseline,
        step_3_establish_targets,
        step_4_coding_outcome,
    )

    current = reshape(coded_path)
    raw = pd.read_csv(coded_path, low_memory=False)
    lihen = raw.loc[
        raw["lemma"].fillna("").str.contains("lîhen", regex=False), "lemma_id"
    ].dropna().unique()
    if len(lihen) != 1:
        raise ValueError(f"Expected one lîhen lemma family, found {lihen.tolist()}")

    forced = raw.copy()
    forced.loc[forced["lemma_id"] == lihen[0], "is_bipartite"] = 1
    forced_path = os.path.join(workdir, "coded_output_lihen_forced_bipartite.csv")
    forced.to_csv(forced_path, index=False)

    normalized = pd.read_csv("data/combined_normalized_corpus.csv", dtype=str)
    processed = step_1_preprocessing(normalized)
    strict_baseline = step_2_establish_baseline(processed, min_anchor_support=2)
    targets = step_3_establish_targets(processed)
    strict_coded = step_4_coding_outcome(processed, strict_baseline, targets)
    strict_path = os.path.join(workdir, "coded_output_min_anchor_support_2.csv")
    strict_coded.to_csv(strict_path, index=False)

    return pd.DataFrame(
        [
            _scenario_row("as coded", current),
            _scenario_row("lîhen forced bipartite in both varieties", reshape(forced_path)),
            _scenario_row("all anchor modes require >=2 agreeing tokens", reshape(strict_path)),
        ]
    )


def print_marking_table(rows, title):
    print(f"\n{title}")
    print(f"  {'marking_type':<22}{'obs':>8}{'leveled':>10}{'rate':>9}")
    for marking, total, lev, rate in rows:
        print(f"  {marking:<22}{total:>8,}{lev:>10,}{rate:>8.2f}%")


def print_periods(long):
    sub = long[(long["marking_type"] == "vowel_bipartite") & long["date"].notna()]
    print("\nBipartite vowel leveling by period")
    print(f"  {'period':<14}{'obs':>8}{'leveled':>10}{'rate':>9}")
    for lo, hi in PERIODS:
        window = sub[(sub["date"] >= lo) & (sub["date"] < hi)]
        total, lev = len(window), int(window["has_levelled"].sum())
        rate = 100 * lev / total if total else 0.0
        warn = "   <- too few observations to read as a rate" if 0 < total < 30 else ""
        print(f"  {lo}-{hi:<9}{total:>8,}{lev:>10,}{rate:>8.2f}%{warn}")


def print_concentration(long):
    sub = long[long["marking_type"] == "vowel_bipartite"]
    total_events = int(sub["has_levelled"].sum())
    by_lemma = (
        sub.groupby(["lemma_id", "lemma"])["has_levelled"]
        .agg(obs="count", leveled="sum")
        .sort_values("leveled", ascending=False)
    )
    print("\nWhere the bipartite vowel events come from")
    print(f"  {'lemma_id':>9}  {'lemma':<16}{'obs':>7}{'leveled':>9}{'share':>9}")
    running = 0
    for (lid, lemma), row in by_lemma.iterrows():
        if row["leveled"] == 0:
            continue
        share = 100 * row["leveled"] / total_events if total_events else 0.0
        running += row["leveled"]
        print(f"  {lid:>9}  {lemma:<16}{int(row['obs']):>7}{int(row['leveled']):>9}{share:>8.1f}%")
    contributing = int((by_lemma["leveled"] > 0).sum())
    print(f"  {'':>9}  {'':<16}{'':>7}{running:>9}{'':>9}")
    print(f"  {total_events} events across {contributing} of {len(by_lemma)} bipartite lemmas.")
    if not by_lemma.empty and total_events:
        top = by_lemma.iloc[0]
        top_share = 100 * top["leveled"] / total_events
        if top_share >= 40:
            print(
                f"  Warning: {by_lemma.index[0][1]} (lemma_id {by_lemma.index[0][0]}) alone supplies "
                f"{top_share:.0f}% of them. The bipartite result is effectively one verb wide; "
                "check that this lemma_id is one etymological family before reading anything into it."
            )


def print_channel_breakdown(coded_path):
    """
    Leveling rates disaggregated by contrast channel.

    The marking_type table above collapses two questions into one with an OR:
    did this past token level to the present stem, and did the past singular and
    plural level to each other. Their base rates differ by an order of
    magnitude, and the mixture of the two differs by marking type, so the
    collapsed rate is not a rate of anything in particular. This table separates
    them.

    Each observation is one source token. Multiple tokens of the same lemma and
    slot in one document remain separate attestations; document_id is used only
    to cluster those observations in the mixed model.
    """
    df = pd.read_csv(coded_path, low_memory=False)
    _validate_token_identity(df, coded_path)
    for col in ("is_leveled_vowel_pres", "is_leveled_vowel_past",
                "is_leveled_cons_pres", "is_leveled_cons_past", "is_bipartite"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["is_bipartite"].notna()]

    print("\nLeveling by contrast channel (disaggregated)")
    print(f"  {'channel':<28}{'marking':<13}{'obs':>8}"
          f"{'leveled':>9}{'rate':>9}")

    channels = [
        ("is_leveled_vowel_past", "vowel (past sg ~ past pl)", (0, 1)),
        ("is_leveled_vowel_pres", "vowel (pres ~ past)", (0, 1)),
        ("is_leveled_cons_past", "cons (past sg ~ past pl)", (1,)),
        ("is_leveled_cons_pres", "cons (pres ~ past)", (1,)),
    ]
    names = {0: "unipartite", 1: "bipartite"}

    for col, label, markings in channels:
        present = df[df[col].notna()]
        for marking in markings:
            obs = present[present["is_bipartite"] == marking]
            n_obs, leveled = len(obs), int(obs[col].sum())
            rate = 100 * leveled / n_obs if n_obs else 0.0
            print(f"  {label:<28}{names[marking]:<13}"
                  f"{n_obs:>8,}{leveled:>9,}{rate:>8.2f}%")


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["marking_type", "observations", "leveled", "leveling_pct"])
        for marking, total, lev, rate in rows:
            writer.writerow([marking, total, lev, f"{rate:.4f}"])
    print(f"\nWrote {path}")


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
                values.append(f"{value:.4g}" if pd.notna(value) else "NA")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_sensitivity(workdir):
    """
    Re-run the coding against a target table with every VARIANT_POLICY choice
    flipped to the rejected variant. Returns the alternative long frame, or None
    if the table has no policy rows.
    """
    from data.corpus_approach_coding import run_pipeline

    rows = list(csv.DictReader(open(TARGETS, encoding="utf-8")))
    flipped = []
    for row in rows:
        if row.get("variant_selection") != "policy":
            continue
        others = [v for v in row["nhg_preterite_variants"].split("|") if v != row["nhg_preterite"]]
        if not others:
            continue
        flipped.append((row["lemma"], row["nhg_infinitive"], row["nhg_preterite"], others[0]))
        row["nhg_preterite"] = others[0]

    if not flipped:
        print("No policy-chosen preterites in the table; nothing to flip.")
        return None

    print("\nFlipping every VARIANT_POLICY choice to the rejected variant:")
    for lemma, inf, was, now in flipped:
        print(f"  {inf:<12} {was:<10} -> {now}")

    alt_targets = os.path.join(workdir, "nhg_targets_flipped.csv")
    alt_coded = os.path.join(workdir, "coded_output_flipped.csv")
    with open(alt_targets, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRe-running the coding pipeline against {alt_targets} ...")
    run_pipeline(output_file=alt_coded, nhg_file=alt_targets)
    return reshape(alt_coded)


def main():
    parser = argparse.ArgumentParser(description="Marking-type summary from coded_output.csv")
    parser.add_argument("--coded", default=CODED, help="coded output to summarise")
    parser.add_argument("--out", default=SUMMARY_OUT, help="marking_type summary CSV")
    parser.add_argument("--report", default=REPORT_OUT, help="generated Markdown report")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="re-run the coding with every VARIANT_POLICY preterite flipped, and compare",
    )
    args = parser.parse_args()

    long = reshape(args.coded)
    print(f"Reshaped {len(long):,} modeling observations across "
          f"{long['lemma_id'].nunique()} lemmas from {args.coded}")
    rows = marking_table(long)
    print_marking_table(rows, "Marking type (Overall / OR-collapsed)")
    model_summary = model_observation_summary(long)
    model_concentration = model_event_concentration(long)
    print("\nExplicit GAMM observations: distinct document x lemma x slot x outcome")
    print(model_summary.to_string(index=False))
    print_channel_breakdown(args.coded)
    print_periods(long)
    print_concentration(long)

    summary = marking_frame(long)
    periods = period_frame(long)
    concentration = concentration_frame(long)
    channels = channel_frame(args.coded)
    os.makedirs(REPORT_DIR, exist_ok=True)
    summary.to_csv(args.out, index=False)
    model_summary.to_csv(f"{REPORT_DIR}/model_observation_summary.csv", index=False)
    model_concentration.to_csv(f"{REPORT_DIR}/model_event_concentration.csv", index=False)
    periods.to_csv(f"{REPORT_DIR}/marking_type_periods.csv", index=False)
    concentration.to_csv(f"{REPORT_DIR}/marking_type_lemma_events.csv", index=False)
    channels.to_csv(f"{REPORT_DIR}/marking_type_channels.csv", index=False)

    with tempfile.TemporaryDirectory(prefix="lihen_sensitivity_") as workdir:
        lihen = run_lihen_sensitivity(args.coded, workdir)
    lihen.to_csv(f"{REPORT_DIR}/lihen_sensitivity.csv", index=False)

    rejected = pd.DataFrame()

    if args.sensitivity:
        with tempfile.TemporaryDirectory(prefix="marking_sensitivity_") as workdir:
            alt = run_sensitivity(workdir)
        if alt is not None:
            alt_rows = marking_table(alt)
            print_marking_table(alt_rows, "Marking type, VARIANT_POLICY choices flipped")
            print("\nDifference (flipped minus current)")
            print(f"  {'marking_type':<22}{'obs':>8}{'leveled':>10}")
            for (marking, t0, l0, _), (_, t1, l1, _) in zip(rows, alt_rows):
                print(f"  {marking:<22}{t1 - t0:>+8,}{l1 - l0:>+10,}")
            rejected = model_summary.merge(
                model_observation_summary(alt),
                on="marking_type",
                suffixes=("_production", "_rejected_variants")
            )
            rejected["observation_difference"] = (
                rejected["observations_rejected_variants"] - rejected["observations_production"]
            )
            rejected["event_difference"] = (
                rejected["leveled_rejected_variants"] - rejected["leveled_production"]
            )
            rejected.to_csv(f"{REPORT_DIR}/rejected_variant_sensitivity.csv", index=False)

    report = f"""# Marking-Type and Target-Variant Audit

Every observation is one source token. The vowel-only model uses the first two
rows of the marking table; the consonant channel is reported separately.

## Marking types

{_markdown(summary)}

These are source-token channel counts. The GAMM deliberately removes repeated
instances of the same outcome within a document-lemma-slot cell. The exact
pre-fit check is:

## Explicit GAMM observations

{_markdown(model_summary)}

## Bipartite events at the GAMM observation unit

{_markdown(model_concentration)}

## Bipartite vowel events by lemma

{_markdown(concentration)}

## lîhen and weak-anchor sensitivity

{_markdown(lihen)}

## Rejected modern variants

{_markdown(rejected) if not rejected.empty else 'Run with `--sensitivity` to regenerate this table.'}

Period and disaggregated contrast counts are stored in
`marking_type_periods.csv` and `marking_type_channels.csv`.

*Generated by `analysis/marking_type_summary.py`.*
"""
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"\nWrote {args.out}, {args.report}, and supporting marking-type tables.")


if __name__ == "__main__":
    main()
