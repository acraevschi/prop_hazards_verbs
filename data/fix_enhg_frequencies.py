#!/usr/bin/env python3
"""
Rescale the ENHG frequency columns in the derived CSVs, in place.

Why
---
extract_enhg_data.py used to divide the ENHG frequency counts by len(results),
the number of extracted verb tokens, while extract_mhg_data.py divides by the
running word count of its corpus. The two `*_freq_per_1000` columns were
therefore on different scales:

    MHG  denominator  2,579,276   all tokens in ReM
    ENHG denominator    262,403   extracted verb tokens only

That made every ENHG figure about 11.84 times larger than the comparable MHG
one. Because corpus is perfectly confounded with date, the offset put a step
change at 1350 into log_freq and log_token_freq, which the GAMM would read as a
frequency effect over time.

extract_enhg_data.py is now fixed at the source. This script exists so that the
correction can be applied to the existing derived files without re-parsing 190
TEI files and re-running the whole chain. Re-running extraction from scratch
produces the same numbers.

What it does
------------
Both pipelines count their numerators over the extracted verb tokens, so only
the denominator was wrong. The repair is one multiplication:

    corrected = current * (OLD_DENOMINATOR / NEW_DENOMINATOR)

applied to lemma_freq_per_1000 and form_freq_per_1000, on ENHG rows only.
lemma_count and form_count are raw counts over the extracted set in both
corpora, so they are already consistent and are left alone.

Safety
------
The script is idempotent. Before writing it recovers the denominator each file
actually implies, as lemma_count * 1000 / lemma_freq_per_1000, and refuses to
act unless that denominator is the old one. A file already corrected is
reported and skipped. A file implying neither denominator is reported and
skipped, because it is not a file this script understands.

Usage
-----
    python3 data/fix_enhg_frequencies.py            # report only
    python3 data/fix_enhg_frequencies.py --apply    # rewrite the files
"""

import argparse
import os

import pandas as pd

# Extracted ENHG verb tokens, the denominator the old code used. Recoverable
# from any uncorrected file as lemma_count * 1000 / lemma_freq_per_1000.
OLD_DENOMINATOR = 262_403

# Every <token> element in data/enhg_corpus/{ref-mlu,ref-rub}, which is what
# the fixed extract_enhg_data.py now counts. Verify with --recount.
NEW_DENOMINATOR = 3_107_196

FREQ_COLUMNS = ["lemma_freq_per_1000", "form_freq_per_1000"]

# Every derived file that carries the ENHG frequency columns, in pipeline order,
# with a flag saying whether the whole file is ENHG. enhg_corpus.csv is, and its
# `corpus` column holds ReF metadata ("ReF.MLU", "Grundkorpus") rather than a
# corpus label, so it must not be filtered on that column. From
# combined_corpus.csv onwards, enhg_mhg_mapping.py writes the label "ENHG".
TARGETS = [
    ("data/enhg_corpus.csv", True),
    ("data/combined_corpus.csv", False),
    ("data/combined_normalized_corpus.csv", False),
    ("data/coded_output.csv", False),
]

ENHG_LABELS = {"ENHG"}


def recount_corpus(folder="data/enhg_corpus", subfolders=("ref-mlu", "ref-rub")):
    """Count every <token> in the subfolders extract_enhg_data.py reads."""
    import xml.etree.ElementTree as ET

    total = 0
    for sub in subfolders:
        path = os.path.join(folder, sub)
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            if not name.endswith(".xml"):
                continue
            root = ET.parse(os.path.join(path, name)).getroot()
            total += len(root.findall(".//token"))
    return total


def enhg_mask(frame, all_enhg):
    """Rows belonging to the ENHG corpus."""
    if all_enhg or "corpus" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["corpus"].isin(ENHG_LABELS)


def implied_denominator(frame, mask):
    """
    The denominator the file's own numbers imply.

    Returns None when the rows disagree, which means the file is not a clean
    product of one extraction run and must not be rescaled blindly.
    """
    sub = frame.loc[mask]
    usable = sub[(sub["lemma_freq_per_1000"] > 0) & (sub["lemma_count"] > 0)]
    if usable.empty:
        return None
    implied = (usable["lemma_count"] * 1000 / usable["lemma_freq_per_1000"]).round(0)
    values = implied.unique()
    if len(values) != 1:
        return None
    return int(values[0])


def process(path, all_enhg, apply_changes):
    if not os.path.exists(path):
        print(f"  {path}: missing, skipped")
        return False

    frame = pd.read_csv(path, low_memory=False)
    missing = [c for c in FREQ_COLUMNS + ["lemma_count"] if c not in frame.columns]
    if missing:
        print(f"  {path}: no frequency columns ({', '.join(missing)}), skipped")
        return False

    mask = enhg_mask(frame, all_enhg)
    if not mask.any():
        print(f"  {path}: no ENHG rows, skipped")
        return False

    denominator = implied_denominator(frame, mask)
    if denominator == NEW_DENOMINATOR:
        print(f"  {path}: already corrected ({mask.sum():,} ENHG rows), skipped")
        return False
    if denominator != OLD_DENOMINATOR:
        found = "inconsistent across rows" if denominator is None else f"{denominator:,}"
        print(
            f"  {path}: implied denominator is {found}, expected {OLD_DENOMINATOR:,}. "
            "Not rescaling - inspect this file by hand."
        )
        return False

    factor = OLD_DENOMINATOR / NEW_DENOMINATOR
    before = frame.loc[mask, "lemma_freq_per_1000"].median()
    if apply_changes:
        for column in FREQ_COLUMNS:
            frame.loc[mask, column] = frame.loc[mask, column] * factor
        frame.to_csv(path, index=False)
    after = before * factor
    verb = "rescaled" if apply_changes else "would rescale"
    print(
        f"  {path}: {verb} {mask.sum():,} ENHG rows by {factor:.8f} "
        f"(median lemma_freq_per_1000 {before:.4f} -> {after:.4f})"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Rescale ENHG frequency columns in place.")
    parser.add_argument("--apply", action="store_true", help="write the files; without it, only report")
    parser.add_argument("--recount", action="store_true", help="re-count the TEI corpus and check NEW_DENOMINATOR")
    args = parser.parse_args()

    if args.recount:
        counted = recount_corpus()
        status = "matches" if counted == NEW_DENOMINATOR else "DOES NOT MATCH"
        print(f"Recounted ENHG corpus: {counted:,} tokens, {status} NEW_DENOMINATOR = {NEW_DENOMINATOR:,}\n")
        if counted != NEW_DENOMINATOR:
            raise SystemExit("Update NEW_DENOMINATOR before applying the correction.")

    print(f"ENHG frequency denominator: {OLD_DENOMINATOR:,} -> {NEW_DENOMINATOR:,} "
          f"(current values are {NEW_DENOMINATOR / OLD_DENOMINATOR:.3f}x too large)")
    print()
    changed = sum(process(path, all_enhg, args.apply) for path, all_enhg in TARGETS)
    print()
    if args.apply:
        print(f"Rewrote {changed} file(s).")
    else:
        print(f"{changed} file(s) would change. Re-run with --apply to write them.")


if __name__ == "__main__":
    main()
