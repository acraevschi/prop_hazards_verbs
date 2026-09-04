#!/usr/bin/env python3
"""
Marking-type summary, straight from data/coded_output.csv.

Why this exists
---------------
The marking_type counts that everybody quotes - 1,622/13, 884/65, and so on - are
computed inside analysis/run_brms.R, which also fits six Stan models and takes
about six hours. So the one table you want after every change to the coding is
locked behind the one step you cannot afford to run, and any number reported
without it is somebody's undocumented re-derivation.

This script does the reshape and nothing else. It is a faithful port of the
"2. Reshaping & Predictor Construction" block of run_brms.R (lines ~270-345):
same vowel_leveled_any / cons_leveled_any collapse, same two filters, same long
pivot, same marking_type construction, same final de-duplication. It writes
nothing the pipeline reads, so it cannot disturb the fits.

If run_brms.R's reshape changes, this port has to change with it. The row count
it reports should equal the "Prepared N modeling observations" line that
run_brms.R prints.

What it prints
--------------
1. The marking_type table: observations, leveling events, rate.
2. Bipartite vowel leveling by period, which is where the S-curve claim lives.
3. Per-lemma contribution to the bipartite vowel events. Read this one. The
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

MARKING_ORDER = ["vowel_unipartite", "vowel_bipartite", "consonant_bipartite"]
PERIODS = [(1050, 1200), (1200, 1350), (1350, 1500), (1500, 1650)]


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


def reshape(coded_path=CODED):
    """Port of run_brms.R's base_model_data -> model_data. Returns the long frame."""
    df = pd.read_csv(coded_path, low_memory=False)

    for col in ("is_leveled_vowel_pres", "is_leveled_vowel_past",
                "is_leveled_cons_pres", "is_leveled_cons_past"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # run_brms.R replaces `lemma` with the shortest surface string per lemma_id
    # before anything else, so every downstream group and the final unique() see
    # that label rather than the raw surface form.
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

    # The predictors have to be carried into the de-duplication key, not just
    # the identifiers. `id` is a document id, not a token id, so unique() in R
    # collapses whole documents; two rows that agree on the identifiers but
    # differ on a frequency predictor survive as two, and dropping those columns
    # here would silently delete observations the model sees.
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
        "id", "variety", "std_infl", "corpus",
    ]
    long = long[keep].drop_duplicates()
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

    Two counts are given for each cell:

      tokens  every coded row, which is what a raw read of coded_output.csv
              gives and what the printed rates used to rest on;
      obs     after the de-duplication run_brms.R performs with unique(). `id`
              is a document id, so that collapses all tokens of one lemma in one
              document that share a date, a slot and an outcome. It is roughly a
              factor of two, it is not uniform across cells, and it is the unit
              the models are actually fitted on.

    Report the obs column. The token column is here so the difference between
    the two is visible rather than a source of quiet disagreement between this
    script and the fits.
    """
    df = pd.read_csv(coded_path, low_memory=False)
    for col in ("is_leveled_vowel_pres", "is_leveled_vowel_past",
                "is_leveled_cons_pres", "is_leveled_cons_past", "is_bipartite"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["is_bipartite"].notna()]

    print("\nLeveling by contrast channel (disaggregated)")
    print(f"  {'channel':<28}{'marking':<13}{'tokens':>8}{'obs':>8}"
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
        # Same key run_brms.R's unique() reduces on, restricted to the columns
        # that identify one observation of this one channel.
        deduped = present.drop_duplicates(
            ["lemma_id", "variety", "corpus", "id", "date", "std_infl", col]
        )
        for marking in markings:
            tokens = present[present["is_bipartite"] == marking]
            obs = deduped[deduped["is_bipartite"] == marking]
            n_obs, leveled = len(obs), int(obs[col].sum())
            rate = 100 * leveled / n_obs if n_obs else 0.0
            print(f"  {label:<28}{names[marking]:<13}{len(tokens):>8,}"
                  f"{n_obs:>8,}{leveled:>9,}{rate:>8.2f}%")


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["marking_type", "observations", "leveled", "leveling_pct"])
        for marking, total, lev, rate in rows:
            writer.writerow([marking, total, lev, f"{rate:.4f}"])
    print(f"\nWrote {path}")


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
    parser.add_argument("--out", help="also write the marking_type table to this CSV")
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
    print_channel_breakdown(args.coded)
    print_periods(long)
    print_concentration(long)

    if args.out:
        write_csv(rows, args.out)

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


if __name__ == "__main__":
    main()
