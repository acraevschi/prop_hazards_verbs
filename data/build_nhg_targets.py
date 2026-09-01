#!/usr/bin/env python3
"""
Assembles the modern German target forms, one row per lemma_id.

The teleological target of leveling is the form the verb reached in modern
German. The corpus cannot supply it: 46 of the 88 modelled lemmas have no late
past-tense attestation, and 64% of the late forms that do exist match neither
historical root. A dictionary form has none of those problems.

Two sources fill the table:
  - The present target is the modern infinitive, taken from the DWDS headword
    already scraped into data/lemmas/enhg_mapping.json, or from a UniMorph lemma
    where DWDS has nothing.
  - The past target comes exclusively from UniMorph (deu), 3rd person singular
    preterite indicative, extracted by data/extract_nhg_preterites.py. Provenance
    and licence live in that script's header.

Choosing the modern infinitive for a lemma family
-------------------------------------------------
A lemma_id is a family of surface variants: lemma 38 holds *finden*, *befinden*,
*empfinden*, *erfinden*. Several of them hit the DWDS mapping, so the family has
to pick one, and the pick decides both targets.

Picking the alphabetically first variant, as this script used to, systematically
returns the prefixed member: *befinden* beats *finden*, *berufen* beats *rufen*,
*erfahren* beats *fahren*, *anfangen* beats *fangen*. That matters in two ways.
A separable prefix gives a preterite with a space in it (*anfangen* -> "fing
an"), which is not the bare form the pipeline expects, and it corrupts the
present target too: extract_root_structure does not know the separable prefix
*an-*, so *anfangen* yields the coda "nf" instead of "ng".

rank_headwords therefore orders the candidates by, in order:

  1. whether UniMorph gives the headword a usable preterite at all;
  2. whether that preterite is a single word - separable verbs are pushed down;
  3. how many members of the family point at this headword - *ziehen* has many,
     the *zeigen* that DSU clustering wrongly pulled into lemma 17 has one;
  4. the shorter string, which is the simplex where a simplex is present;
  5. the string itself, so the result does not move between runs.

Inseparable prefixes that survive this (*vergessen*, *verlieren*, *geschehen*)
are harmless: extract_root_structure strips ver-, ge-, be-, er-, ent- before it
reads the root, so the vowel and coda match the simplex.

Output: data/lemmas/nhg_targets.csv, plus data/lemmas/nhg_targets_misses.csv
recording every lemma the source could not cover and what was tried for it.
"""

import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.extract_nhg_preterites import SOURCE_TAG, load_mapping

OUT = "data/lemmas/nhg_targets.csv"
MISSES = "data/lemmas/nhg_targets_misses.csv"


def clean_for_query(lemma):
    """Normalization to extract core verb stem from corpus lemma annotations."""
    if not lemma or not isinstance(lemma, str):
        return ""
    text = re.sub(r"[()]", "", lemma).strip()
    text = text.split("/")[0].strip()
    text = text.split("-")[-1].strip()
    parts = [p.strip() for p in text.split("|") if p.strip()]
    return parts[1] if len(parts) > 1 else (parts[0] if parts else "")


def strip_diacritics(text):
    """Drops the MHG length and quality marks, so brâten matches braten."""
    if not text:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def query_forms(form):
    """The spellings under which one surface variant is looked up, in order."""
    seen = []
    for candidate in (form, clean_for_query(form), strip_diacritics(clean_for_query(form))):
        if candidate and candidate not in seen:
            seen.append(candidate)
    return seen


def rank_headwords(votes, preterites):
    """
    Order candidate infinitives best-first. See the module docstring for why.

    `votes` maps a candidate infinitive to the number of family variants that
    reached it. `preterites` is the UniMorph mapping.
    """
    def key(headword):
        entry = preterites.get(headword) or {}
        preterite = entry.get("preterite", "")
        return (
            0 if preterite else 1,
            1 if " " in preterite else 0,
            -votes[headword],
            len(headword),
            headword,
        )

    return sorted(votes, key=key)


def main():
    # The lemma list comes from the normalized corpus, the last artifact before
    # the coding step. It used to come from coded_output.csv, which the coding
    # step produces from this very table - a cycle that is invisible until the
    # lemma families are relinked. Then coded_output.csv still carries the old
    # lemma_ids while lemma_id.csv carries the new ones, and every shifted id
    # gets the wrong family's spellings. Reading the corpus instead puts this
    # script cleanly between normalize_data.py and corpus_approach_coding.py.
    corpus_path = "data/combined_normalized_corpus.csv"
    lemma_id_path = "data/lemmas/lemma_id.csv"
    analysis_path = "analysis/data_for_analysis.csv"

    corpus = []
    if os.path.exists(corpus_path):
        corpus = list(csv.DictReader(open(corpus_path, encoding="utf-8")))
    else:
        raise SystemExit(
            f"{corpus_path} is missing. Run data/normalize_data.py before this script."
        )

    # `in_model` is an audit label, not an input: it says whether the last
    # completed run of analysis/run_brms.R kept this lemma. It matches on family
    # membership - does any surface variant of this family appear in that run -
    # for two reasons. lemma_std there is a lemma_id, and lemma_ids are
    # positional: data/lemmas/enhg_mhg_mapping.py numbers the DSU components in
    # sorted order, so splitting one family shifts the id of every family after
    # it, and matching on the id would relabel hundreds of rows after a relink.
    # The `lemma` column there is no better on its own, because run_brms.R picks
    # its representative from the coded tokens while this script picks one from
    # the whole family, and the two need not agree.
    model_lemmas = set()
    if os.path.exists(analysis_path):
        model_lemmas = {
            r["lemma"]
            for r in csv.DictReader(open(analysis_path, encoding="utf-8"))
        }

    # Collect all surface variants per lemma_id
    spellings = {}
    if os.path.exists(lemma_id_path):
        for row in csv.DictReader(open(lemma_id_path, encoding="utf-8")):
            if row.get("lemma") and row.get("lemma_id"):
                spellings.setdefault(row["lemma_id"], set()).add(row["lemma"])

    for row in corpus:
        if row.get("lemma") and row.get("lemma_id"):
            spellings.setdefault(row["lemma_id"], set()).add(row["lemma"])

    # Load DWDS headwords mapping
    headwords = {}
    dwds_mapping_path = "data/lemmas/enhg_mapping.json"
    if os.path.exists(dwds_mapping_path):
        with open(dwds_mapping_path, encoding="utf-8") as handle:
            headwords = json.load(handle)

    preterites = load_mapping()

    def find_headword(forms):
        """
        Resolve a lemma family to one modern infinitive.

        Returns (infinitive, source). DWDS is tried first for the whole family,
        UniMorph lemmas only if DWDS reached nothing, so the two are never mixed
        inside one decision.
        """
        for source, table in (("dwds", headwords), ("unimorph", preterites)):
            votes = Counter()
            exact = set()
            for form in sorted(forms):
                for candidate in query_forms(form):
                    if candidate in table:
                        resolved = table[candidate] if source == "dwds" else candidate
                        votes[resolved] += 1
                        if candidate == form:
                            exact.add(resolved)
                        break
            if votes:
                best = rank_headwords(votes, preterites)[0]
                if source == "dwds":
                    return best, "dwds" if best in exact else "dwds_normalized"
                return best, "unimorph"

        return "", ""

    corpus_lids = sorted({r["lemma_id"] for r in corpus if r.get("lemma_id")}, key=int)

    rows = []
    misses = []
    for lemma_id in corpus_lids:
        forms = spellings.get(lemma_id, set())
        representative = sorted(forms, key=lambda s: (len(s), s))[0] if forms else ""
        in_model = "yes" if forms & model_lemmas else "no"
        infinitive, infinitive_source = find_headword(forms)

        entry = preterites.get(infinitive) or {}
        preterite = entry.get("preterite", "")
        variants = entry.get("variants", [])
        selection = entry.get("selection", "")
        note = entry.get("note", "")

        if selection == "unresolved":
            # Reachable only if a lemma the table needs acquires several
            # UniMorph preterites. File order must not decide it silently.
            raise SystemExit(
                f"lemma_id {lemma_id} ({representative}) resolves to {infinitive!r}, for which "
                f"UniMorph lists several preterites {variants} and no VARIANT_POLICY entry "
                f"exists.\nAdd {infinitive!r} to VARIANT_POLICY in data/extract_nhg_preterites.py "
                "with a reason, then re-run that script."
            )

        if preterite:
            preterite_source = SOURCE_TAG
            confidence = "policy" if selection == "policy" else "high"
        else:
            preterite_source = ""
            confidence = ""
            reason = (
                f"no UniMorph entry for the resolved infinitive {infinitive!r}"
                if infinitive
                else "no modern infinitive found in DWDS or UniMorph"
            )
            note = reason
            misses.append(
                {
                    "lemma_id": lemma_id,
                    "lemma": representative,
                    "in_model": in_model,
                    "nhg_infinitive": infinitive,
                    "reason": reason,
                    "spellings_tried": "|".join(sorted(forms)),
                }
            )

        rows.append(
            {
                "lemma_id": lemma_id,
                "lemma": representative,
                "in_model": in_model,
                "nhg_infinitive": infinitive,
                "nhg_preterite": preterite,
                "nhg_preterite_variants": "|".join(variants),
                "variant_selection": selection,
                "confidence": confidence,
                "source_infinitive": infinitive_source,
                "source_preterite": preterite_source,
                "curator_note": note,
            }
        )

    # Rows in the model come first, incomplete first, then by lemma
    rows.sort(
        key=lambda r: (
            r["in_model"] != "yes",
            bool(r["nhg_infinitive"]) and bool(r["nhg_preterite"]),
            r["lemma"],
        )
    )

    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    misses.sort(key=lambda r: (r["in_model"] != "yes", r["lemma"]))
    with open(MISSES, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["lemma_id", "lemma", "in_model", "nhg_infinitive", "reason", "spellings_tried"],
        )
        writer.writeheader()
        writer.writerows(misses)

    in_model = [r for r in rows if r["in_model"] == "yes"]
    by_policy = [r for r in rows if r["variant_selection"] == "policy"]
    print(f"Wrote {len(rows)} rows to {OUT}")
    print(f"  lemmas in the model            : {len(in_model)}")
    print(f"  with a modern infinitive       : {sum(1 for r in in_model if r['nhg_infinitive'])}")
    print(f"  with a modern preterite        : {sum(1 for r in in_model if r['nhg_preterite'])}")
    print(f"  complete                       : {sum(1 for r in in_model if r['nhg_infinitive'] and r['nhg_preterite'])}")
    print(f"  preterite chosen by policy     : {len(by_policy)} ({sum(1 for r in by_policy if r['in_model'] == 'yes')} in the model)")
    for row in sorted(by_policy, key=lambda r: r["nhg_infinitive"]):
        rejected = [v for v in row["nhg_preterite_variants"].split("|") if v != row["nhg_preterite"]]
        flag = "*" if row["in_model"] == "yes" else " "
        print(f"    {flag} {row['nhg_infinitive']:<12} -> {row['nhg_preterite']:<10} (over {', '.join(rejected)})")

    print(f"\n  no preterite from the source   : {len(misses)} ({sum(1 for m in misses if m['in_model'] == 'yes')} in the model)")
    print(f"  listed in {MISSES}")
    for miss in misses:
        if miss["in_model"] == "yes":
            print(f"    {miss['lemma']}: {miss['reason']}")


if __name__ == "__main__":
    main()
