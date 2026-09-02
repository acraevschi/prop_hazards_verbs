#!/usr/bin/env python3
"""
Builds the draft curation table for teleological targets.

The target of paradigm leveling is not a free-form vowel. Leveling makes one
paradigm cell adopt the root of another cell, so the endpoint of past-tense
leveling is one of two things: the past singular anchor, or the past plural
anchor. This script writes one row per lemma with the evidence a curator needs
to record which of the two the modern verb continues.

The script proposes an answer from the corpus, but it does not decide. The late
ENHG evidence is thin, so a curator confirms or corrects each row against the
modern form, and the confirmed file is the input to the coding pipeline.

Output: data/lemmas/target_choices_draft.csv
"""

import json
import os
import sys

import pandas as pd
from tqdm import tqdm

# Run from the repository root. The paths below are relative to it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_approach_coding as cac

LATE_CUTOFF = 1500  # earliest date counted as late ENHG evidence
MIN_COUNT = 3  # fewest classified tokens for a corpus proposal
MIN_MARGIN = 2.0  # winning pile must be this many times the losing pile


def classify_against_anchors(forms, anchor_self, anchor_other, is_cons, variety, sc_dict):
    """
    Sorts observed roots into three piles: the ones that match anchor_self, the
    ones that match anchor_other, and the ones that match neither.

    The third pile is the important one. A root that matches neither anchor
    cannot be the endpoint of leveling, so an open-ended mode over these forms
    is what puts impossible targets into the data.
    """

    def equivalent(a, b):
        if is_cons:
            return cac.are_cons_equivalent(a, b)
        return cac.are_vowels_equivalent(a, b, variety, sc_dict)

    hits_self, hits_other, unmatched = 0, 0, 0
    for form in forms:
        if pd.isna(form):
            continue
        match_self = not pd.isna(anchor_self) and equivalent(form, anchor_self)
        match_other = not pd.isna(anchor_other) and equivalent(form, anchor_other)
        if match_self and not match_other:
            hits_self += 1
        elif match_other and not match_self:
            hits_other += 1
        elif match_self and match_other:
            # The two anchors are not distinguishable here. The form carries no
            # information about direction, so it counts for neither pile.
            unmatched += 1
        else:
            unmatched += 1
    return hits_self, hits_other, unmatched


def propose(hits_sg, hits_pl):
    """Names the winning anchor, or 'unresolved' when the evidence is thin."""
    total = hits_sg + hits_pl
    if total < MIN_COUNT:
        return "unresolved", "fewer than %d classified tokens" % MIN_COUNT
    high, low = max(hits_sg, hits_pl), min(hits_sg, hits_pl)
    if low > 0 and high < MIN_MARGIN * low:
        return "unresolved", "no clear majority (%d vs %d)" % (hits_sg, hits_pl)
    return ("pastsg" if hits_sg > hits_pl else "pastpl"), "corpus majority %d vs %d" % (
        hits_sg,
        hits_pl,
    )


def main():
    df = pd.read_csv("data/combined_normalized_corpus.csv", dtype=str)
    processed = cac.step_1_preprocessing(df)
    baseline = cac.step_2_establish_baseline(processed)
    sc_dict = cac.load_sound_changes("data/vowel_changes.csv")

    processed["date_num"] = pd.to_numeric(processed["date"], errors="coerce")
    baseline["lemma_id"] = baseline["lemma_id"].astype(str)
    processed["lemma_id"] = processed["lemma_id"].astype(str)

    headwords = {}
    headword_path = "data/lemmas/enhg_mapping.json"
    if os.path.exists(headword_path):
        with open(headword_path, encoding="utf-8") as handle:
            headwords = json.load(handle)

    # A decision is only needed where the two past anchors really differ. Where
    # the singular and the plural share a root, there is no direction to choose.
    coded_path = "data/coded_output.csv"
    needs_vowel, needs_coda, in_model = set(), set(), set()
    if os.path.exists(coded_path):
        coded = pd.read_csv(coded_path, dtype=str)
        needs_vowel = set(
            coded.loc[coded["diff_vowel_pastsg_pastpl"] == "True", "lemma_id"].astype(str)
        )
        needs_coda = set(
            coded.loc[coded["diff_cons_pastsg_pastpl"] == "True", "lemma_id"].astype(str)
        )
    model_path = "analysis/data_for_analysis.csv"
    if os.path.exists(model_path):
        in_model = set(pd.read_csv(model_path, dtype=str)["lemma_std"].astype(str))

    reference = {}
    reference_path = "data/lemmas/target_choices_reference.csv"
    if os.path.exists(reference_path):
        ref = pd.read_csv(reference_path, dtype=str).fillna("")
        reference = {str(r["lemma_id"]): r for _, r in ref.iterrows()}

    rows = []
    for lemma_id, group in tqdm(processed.groupby("lemma_id"), desc="Building draft"):
        surface = sorted({str(x) for x in group["lemma"].dropna()}, key=len)
        if not surface:
            continue
        representative = surface[0]

        late = group[
            (group["corpus"] == "ENHG")
            & (group["date_num"] >= LATE_CUTOFF)
            & (group["std_infl"].isin(["PastSg", "PastPl"]))
        ]

        # Evidence is pooled over varieties for the proposal, because the
        # curated answer is one direction per lemma. The vowel written into the
        # coded data still comes from each variety's own anchor.
        vowel_sg = vowel_pl = vowel_none = 0
        coda_sg = coda_pl = coda_none = 0

        # Anchors come from the baseline, for every variety this lemma has.
        # They must not depend on whether late forms exist, because a lemma
        # with no late attestation still needs its anchors shown to the curator.
        anchor_rows = baseline[baseline["lemma_id"] == str(lemma_id)]
        anchors = {r["variety"]: r for _, r in anchor_rows.iterrows()}

        for variety, block in late.groupby("variety"):
            anchor_row = anchors.get(variety)
            if anchor_row is None:
                continue

            a, b, c = classify_against_anchors(
                block["extracted_vowel"],
                anchor_row.get("anchor_vowel_pastsg"),
                anchor_row.get("anchor_vowel_pastpl"),
                False,
                variety,
                sc_dict,
            )
            vowel_sg, vowel_pl, vowel_none = vowel_sg + a, vowel_pl + b, vowel_none + c

            a, b, c = classify_against_anchors(
                block["extracted_coda"],
                anchor_row.get("anchor_coda_pastsg"),
                anchor_row.get("anchor_coda_pastpl"),
                True,
                variety,
                sc_dict,
            )
            coda_sg, coda_pl, coda_none = coda_sg + a, coda_pl + b, coda_none + c

        vowel_choice, vowel_note = propose(vowel_sg, vowel_pl)
        coda_choice, coda_note = propose(coda_sg, coda_pl)

        def per_variety(field):
            """Shows the anchor for each variety, so a split stays visible."""
            parts = []
            for variety in sorted(anchors):
                value = anchors[variety].get(field)
                short = "UG" if variety.startswith("Upper") else "CG"
                parts.append(f"{short}:{'-' if pd.isna(value) else value}")
            return " ".join(parts)
        key = str(lemma_id)
        ref = reference.get(key)
        rows.append(
            {
                "lemma_id": lemma_id,
                "lemma": representative,
                "in_model": "yes" if key in in_model else "no",
                "decision_needed_vowel": "yes" if key in needs_vowel else "no",
                "decision_needed_coda": "yes" if key in needs_coda else "no",
                "surface_variants": " | ".join(surface[:6]),
                "dwds_headword": headwords.get(representative, ""),
                "anchor_vowel_pres": per_variety("anchor_vowel_pres"),
                "anchor_vowel_pastsg": per_variety("anchor_vowel_pastsg"),
                "anchor_vowel_pastpl": per_variety("anchor_vowel_pastpl"),
                "anchor_coda_pastsg": per_variety("anchor_coda_pastsg"),
                "anchor_coda_pastpl": per_variety("anchor_coda_pastpl"),
                "late_vowel_sg": vowel_sg,
                "late_vowel_pl": vowel_pl,
                "late_vowel_unmatched": vowel_none,
                "late_coda_sg": coda_sg,
                "late_coda_pl": coda_pl,
                "late_coda_unmatched": coda_none,
                "proposed_vowel_past_winner": vowel_choice,
                "proposed_vowel_note": vowel_note,
                "proposed_coda_past_winner": coda_choice,
                "proposed_coda_note": coda_note,
                # Proposed by reference, for the curator to confirm or correct.
                "nhg_preterite": ref["nhg_preterite"] if ref is not None else "",
                "proposed_winner_reference": ref["proposed_winner"] if ref is not None else "",
                "confidence": ref["confidence"] if ref is not None else "",
                "reference_note": ref["reference_note"] if ref is not None else "",
                # The curator fills these three columns.
                "vowel_past_winner": "",
                "coda_past_winner": "",
                "curator_note": "",
            }
        )

    out = pd.DataFrame(rows)
    # Put the rows that need a decision first, so the curator can stop early.
    out["_work"] = (
        (out["in_model"] == "yes")
        & ((out["decision_needed_vowel"] == "yes") | (out["decision_needed_coda"] == "yes"))
    )
    out = out.sort_values(["_work", "lemma"], ascending=[False, True]).drop(columns="_work")
    path = "data/lemmas/target_choices_draft.csv"
    out.to_csv(path, index=False)

    work = out[
        (out["in_model"] == "yes")
        & ((out["decision_needed_vowel"] == "yes") | (out["decision_needed_coda"] == "yes"))
    ]
    print(f"\nWrote {len(out)} rows to {path}")
    print(f"  rows in the model                    : {(out['in_model'] == 'yes').sum()}")
    print(f"  rows that need a decision            : {len(work)}")
    print(f"  of those, with a reference proposal  : {(work['nhg_preterite'] != '').sum()}")
    counts = work["confidence"].replace("", "none").value_counts().to_dict()
    print(f"  confidence of the proposals          : {counts}")


if __name__ == "__main__":
    main()
