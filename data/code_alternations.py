import argparse
import json
import os
import re
import tempfile
from collections import Counter
from itertools import zip_longest
from typing import Optional, Tuple

import pandas as pd
import warnings
from lingpy import Multiple
from tqdm import tqdm

tqdm.pandas()

# Suppress warnings from lingpy
warnings.filterwarnings("ignore", module="lingpy")

### To disable lingpy messages
import logging

for logger_name in logging.Logger.manager.loggerDict:
    if "lingpy" in logger_name.lower():
        logging.getLogger(logger_name).setLevel(logging.ERROR)

# ---------------------------
# Constants (same as original)
# ---------------------------
VOWEL_NORM_DICT = {
    "â": "a",
    "ā": "a",
    "ā": "a",
    "á": "a",
    "ê": "e",
    "î": "i",
    "ô": "o",
    "û": "u",
    "æ": "ä",
    "œ": "ö",
    "ē": "e",
    "ī": "i",
    "í": "i",
    "ō": "o",
    "oͮ": "o",
    "ū": "u",
    "ū": "u",
    "uͦ": "u",
    "u͗": "u",
    "oͤ": "o",
    "eͤ": "e",
    "aͤ": "a",
    "vͦ": "ü",
    "v͗": "ü",
    "v̈": "ü",
    "ÿ": "i",
    "y": "i",
    "ë": "e",
    "è": "e",
    "é": "e",
    "ē": "e",
    "ᵉ": "e",
    "ë": "e",
    "eͦ": "o",
}

CONS_NORM_DICT = {"n̄": "n", "w͗": "w", "ß": "s", "ſ": "s", "ˢ": "", "h̄": "h", "t̄": "t"}

DIGRAPH_IPA = {
    "sch": "ʃ",  # German 'sch' → [ʃ]
    "ch": "χ",  # voiceless uvular fricative [χ]
    "ck": "k",  # [k]
    "ph": "f",  # Greek loanwords: [f]
    "pf": "f",
    "th": "t",  # [t]
    "tz": "ʦ",  # affricate [ʦ]
    "ts": "ʦ",  # affricate [ʦ]
    "sz": "s",  # mostly a spelling difference
    "cz": "z",  # spelling alternation
    "vb": "üb",
    "vn": "un",
}

allowed_vf = {("v", "f"), ("f", "v")}
devoicing_pairs = {
    ("b", "p"),
    ("p", "b"),
    ("d", "t"),
    ("t", "d"),
    ("g", "k"),
    ("k", "g"),
    ("z", "s"),
    ("s", "z"),
    ("ʒ", "ʃ"),
    ("ʃ", "ʒ"),
    ("ʤ", "ʧ"),
    ("ʧ", "ʤ"),
    # ("g", "ch") # In bavarian, check an ex later
}

MHG_SUFFIXES = [
    "ende",
    "ente",
    "ent",
    "nde",
    "nden",
    "nte",
    "nten",
    "enne",
    "ent",
    "est",
    "en",
    "et",
    "es",
    "st",
    "t",
    "e",
    "n",
]
ENHG_SUFFIXES = ["ende", "ente", "end", "est", "en", "et", "es", "st", "t", "e", "n"]

### Think how to prettify prefixes' treatment
SEPARABLE_PREFIXES = {
    "MHG": [
        "auf",
        "aus",
        "ein",
        "mit",
        "nach",
        "vor",
        "vur",
        "zer",
        "durch",
        "wider",
        "ent",
        "enʦ",
        "int",
        "en",
        "er",
        "unter",
        "vnter",
        "under",
        "umbe",
        "ab",
        "an",
        "zu",
    ],
    "ENHG": [
        "zer",
        "durch",
        "wider",
        "ent",
        "enʦ",
        "int",
        "en",
        "unter",
        "under",
        "vnter",
        "umbe",
        "auf",
        "aus",
        "ein",
        "mit",
        "nach",
        "vor",
        "vur",
        "ab",
        "an",
        "zu",
        "er",
    ],
}

INSEPARABLE_PREFIXES = {
    "MHG": [
        "unter",
        "under",
        "vnter",
        "über",
        "be",
        "ent",
        "enʦ" "er",
        "miss",
        "umbe",
        "ver",
        "zer",
        "vs",
        "v",
        "ge",
        "g",
        "er",
    ],
    "ENHG": [
        "unter",
        "under",
        "vnter",
        "über",
        "umbe",
        "be",
        "ent",
        "enʦ",
        "er",
        "miss",
        "ver",
        "vor",
        "zer",
        "ge",
        "g",
        "er",
    ],
}
_VOWELS = "aeiouäöüy"

# ---------------------------
# Helper functions (reused & adapted)
# ---------------------------


def strip_common_affixes(
    form: str,
    lemma: Optional[str] = None,
    base_lemma: Optional[str] = None,
    corpus: str = "MHG",
    *,
    normalize_uvj: bool = False,
) -> str:
    """
    Strip common prefixes/suffixes from MHG/ENHG forms with:
      1) vowel/consonant normalization (vowel mapping + digraph-to-IPA)
      2) prefix stripping (separable, inseparable, participle)
      3) special corrections (e.g. 'ffen'→'fen')
      4) suffix stripping with validation.
    """
    if corpus not in ("MHG", "ENHG"):
        raise ValueError("corpus must be 'MHG' or 'ENHG'")
    if form is None:
        return form

    # ----------- 1) NORMALIZE VOWELS / CONSONANTS -----------
    f = form.lower()
    if normalize_uvj:
        f = f.replace("v", "u").replace("j", "i")

    # Replace historical vowel marks
    for src, tgt in VOWEL_NORM_DICT.items():
        f = f.replace(src, tgt)

    for src, tgt in CONS_NORM_DICT.items():
        f = f.replace(src, tgt)

    # Collapse digraphs to IPA (sort by length to avoid partial overlaps)
    for digraph, ipa in sorted(
        DIGRAPH_IPA.items(), key=lambda x: len(x[0]), reverse=True
    ):
        f = f.replace(digraph, ipa)

    def has_vowel(s: str) -> bool:
        return any(v in s for v in _VOWELS)

    # ----------- 2) PREFIX STRIPPING -----------
    f_pref = f
    if lemma and "-" in lemma:
        possible = lemma.split("-")[0]
        if possible in SEPARABLE_PREFIXES[corpus] and f_pref.startswith(possible):
            f_pref = f_pref[len(possible) :]

    for pref in INSEPARABLE_PREFIXES[corpus]:
        if f_pref.startswith(pref) and not (base_lemma and base_lemma.startswith(pref)):
            # needs special treatment, as in some cases we merge "ent" with "s" from stem in some cases
            if pref != "enʦ":
                remainder = f_pref[len(pref) :]
                if has_vowel(remainder):
                    f_pref = remainder
                    break
            # needs special treatment, as in some cases we merge "ent" with "s" from stem in some cases
            elif pref == "enʦ":
                remainder = "s" + f_pref[len(pref) :]
                if has_vowel(remainder):
                    f_pref = remainder
                    break

    # ----------- 3) OTHER HIGHLIGHTED CORRECTIONS -----------
    f_pref = re.sub(r"([b-df-hj-np-tv-z])\1+", r"\1", f_pref)

    # ----------- 4) SUFFIX STRIPPING -----------
    suffixes = MHG_SUFFIXES if corpus == "MHG" else ENHG_SUFFIXES
    for suf in sorted(suffixes, key=len, reverse=True):
        if f_pref.endswith(suf):
            cand = f_pref[: -len(suf)]
            if has_vowel(cand) and len(cand) >= 2:
                f_pref = cand
                break

    return f_pref.strip("- ")


def align_pair(stem: str, base_stem: str, gop: int = -3) -> Optional[Tuple[list, list]]:
    """
    Align two sequences (character lists) using lingpy.Multiple and return two aligned lists.
    If alignment cannot be obtained, return None.
    """
    if stem is None or base_stem is None:
        return None
    # If either is empty, nothing to align
    if not isinstance(stem, str) or not isinstance(base_stem, str):
        return None
    if len(stem) == 0 or len(base_stem) == 0:
        return None

    seq1 = list(stem)
    seq2 = list(base_stem)
    # Use LingPy Multiple to perform an alignment and then parse its printed output.
    temp = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False)
    try:
        mult = Multiple([seq1, seq2])
        mult.prog_align(gop=gop)
        print(mult, file=temp)
        temp.close()

        with open(temp.name, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        sequences = []
        # Parse printed alignment robustly: remove initial line labels like "0:" or "1"
        for line in lines:
            s = re.sub(r"^\s*\d+\s*[:\)]?\s*", "", line)
            s = s.strip()
            if not s:
                continue
            # If tokens are whitespace separated (e.g. "h a l s"), keep them; otherwise split into chars.
            if " " in s:
                tokens = [t for t in s.split() if t not in (".", "|")]
            else:
                tokens = [ch for ch in s if ch not in (".", "|")]
            if tokens:
                sequences.append(tokens)

        if len(sequences) < 2:
            return None
        # Typically the first two non-empty sequence lines are our aligned sequences
        return sequences[0], sequences[1]
    except Exception:
        return None
    finally:
        try:
            os.unlink(temp.name)
        except Exception:
            pass


def trim_aligned_seqs(pair: Tuple[list, list]) -> Tuple[list, list]:
    """
    Remove leading and trailing columns where either sequence has a gap character '-'.
    Expects pair == (aligned_seq1, aligned_seq2)
    Returns trimmed lists.
    """
    stem, base_stem = pair
    if stem is None or base_stem is None:
        return [], []

    # normalize gap character representations (treat '.' as gap too)
    def norm_gap(x):
        return "-" if x == "." else x

    stem = [norm_gap(c) for c in stem]
    base_stem = [norm_gap(c) for c in base_stem]

    leading_gaps = 0
    for a, b in zip(stem, base_stem):
        if a == "-" or b == "-":
            leading_gaps += 1
        else:
            break

    trailing_gaps = 0
    for a, b in zip(reversed(stem), reversed(base_stem)):
        if a == "-" or b == "-":
            trailing_gaps += 1
        else:
            break

    new_stem = stem[
        leading_gaps : len(stem) - trailing_gaps if trailing_gaps > 0 else None
    ]
    new_base_stem = base_stem[
        leading_gaps : len(base_stem) - trailing_gaps if trailing_gaps > 0 else None
    ]

    if len(new_stem) < 2 or len(new_base_stem) < 2:
        return [], []

    return new_stem, new_base_stem


def extract_vowels_consonants_from_aligned(seqa: list, seqb: list):
    """
    From two aligned sequences (lists of characters with possible gaps '-'), produce
    four underscore-joined strings:
      - vowels_stem, vowels_base_stem, cons_stem, cons_base_stem.

    If any of the four lists would be empty, returns None (as in your original logic).
    """
    vowels_stem = []
    vowels_base_stem = []
    cons_stem = []
    cons_base_stem = []

    for a, b in zip_longest(seqa, seqb, fillvalue="-"):
        # a side
        if a != "-":
            if a in _VOWELS:
                vowels_stem.append(a)
            else:
                cons_stem.append(a)
        else:
            # a is gap; if b is non-gap record gap in the corresponding list
            if b != "-":
                if b in _VOWELS:
                    vowels_stem.append("-")
                else:
                    cons_stem.append("-")
        # b side
        if b != "-":
            if b in _VOWELS:
                vowels_base_stem.append(b)
            else:
                cons_base_stem.append(b)
        else:
            if a != "-":
                if a in _VOWELS:
                    vowels_base_stem.append("-")
                else:
                    cons_base_stem.append("-")

    # If any list is empty, return None (as per original)
    if not vowels_stem or not vowels_base_stem or not cons_stem or not cons_base_stem:
        return None

    return {
        "vowels_stem": "_".join(vowels_stem),
        "vowels_base_stem": "_".join(vowels_base_stem),
        "cons_stem": "_".join(cons_stem),
        "cons_base_stem": "_".join(cons_base_stem),
    }


def detect_alternations(vowels_stem, vowels_base_stem, cons_stem, cons_base_stem):
    """
    Detect vowel and consonant alternations between two stems.

    Rules:
    - Vowel alternation: if vowels match exactly -> "no", else "yes".
    - Consonant alternation:
        * If consonant sequences match exactly -> "no".
        * v/f alternation is acceptable anywhere.
        * Devoicing pairs are acceptable only in the final position.
        * All other differences count as alternation.
    """
    vowel_alternation = "no" if vowels_stem == vowels_base_stem else "yes"

    cons_list = cons_stem.split("_") if cons_stem else []
    cons_base_list = cons_base_stem.split("_") if cons_base_stem else []

    if cons_list == cons_base_list:
        return vowel_alternation, "no"

    n = max(len(cons_list), len(cons_base_list))
    # iterate zipped, but if lengths differ treat missing as mismatch (counts as alternation)
    for i in range(n):
        c1 = cons_list[i] if i < len(cons_list) else None
        c2 = cons_base_list[i] if i < len(cons_base_list) else None
        if c1 == c2:
            continue
        if (c1, c2) in allowed_vf:
            continue
        # devoicing allowed only in last position
        if i == n - 1 and (c1, c2) in devoicing_pairs:
            continue
        # Otherwise → alternation
        return vowel_alternation, "yes"

    return vowel_alternation, "no"


# ---------------------------
# Orchestration for one row
# ---------------------------


def compute_row_alternations(row):
    """
    Given a dataframe row with 'stem' and 'base_stem' strings, return (vowel_alternation, cons_alternation)
    or (None, None) if undecidable.
    """
    try:
        check_chars = ["③", "①"]
        for char in check_chars:
            if char in row["stem"] or char in row["base_stem"]:
                return pd.Series({"vowel_alternation": None, "cons_alternation": None})

        aligned = align_pair(row["stem"], row["base_stem"])

        if not aligned:
            return pd.Series({"vowel_alternation": None, "cons_alternation": None})

        trimmed_a, trimmed_b = trim_aligned_seqs(aligned)
        if not trimmed_a or not trimmed_b:
            return pd.Series({"vowel_alternation": None, "cons_alternation": None})

        vc = extract_vowels_consonants_from_aligned(trimmed_a, trimmed_b)
        if vc is None:
            return pd.Series({"vowel_alternation": None, "cons_alternation": None})

        vowel_alt, cons_alt = detect_alternations(
            vc["vowels_stem"],
            vc["vowels_base_stem"],
            vc["cons_stem"],
            vc["cons_base_stem"],
        )

        return pd.Series({"vowel_alternation": vowel_alt, "cons_alternation": cons_alt})
    except Exception:
        # In case of any unexpected error per-row, return Nones (safe fallback)
        return pd.Series({"vowel_alternation": None, "cons_alternation": None})


# ---------------------------
# Main processing pipeline
# ---------------------------


def process_file(input_csv: str, output_csv: Optional[str] = None):
    # Resolve output filename
    base, ext = os.path.splitext(input_csv)
    if ext:
        default_out = f"{base}_coded{ext}"
    else:
        default_out = f"{input_csv}_coded"
    output_csv = output_csv or default_out

    print(f"Reading input: {input_csv}")
    data = pd.read_csv(input_csv, dtype=str)

    # original script cast principal_part to int - do safe conversion
    if "principal_part" in data.columns:
        data["principal_part"] = pd.to_numeric(data["principal_part"], errors="coerce")
    # remove duplicates by norm and id (like your original)
    if {"norm", "id"}.issubset(data.columns):
        data.drop_duplicates(subset=["norm", "id"], inplace=True)

    # Load lemma maps (paths kept as in original)
    with open("data/lemmas/mhg_mapping.json", encoding="utf-8") as f:
        mhg_map = json.load(f)
    with open("data/lemmas/enhg_mapping.json", encoding="utf-8") as f:
        enhg_map = json.load(f)

    # Map base_lemma depending on corpus
    def map_base_lemma(r):
        corpus = r.get("corpus")
        lemma = r.get("lemma")
        if corpus == "MHG":
            return mhg_map.get(lemma)
        else:
            return enhg_map.get(lemma)

    data["base_lemma"] = data.apply(map_base_lemma, axis=1)
    data.dropna(subset=["base_lemma"], inplace=True)

    # Normalization pipeline (kept as your original)
    data["form_normalized"] = data["norm"].str.replace("ſ", "s", regex=False)
    data["form_normalized"] = data["form_normalized"].str.replace(
        r"(.)\1+", r"\1", regex=True
    )
    data["form_normalized"] = data["form_normalized"].str.replace(
        r"[\[\]\(\)\{\}]", "", regex=True
    )
    data["form_normalized"] = data["form_normalized"].str.replace(".", "", regex=False)
    data["form_normalized"] = data["form_normalized"].str.replace(
        "\u200d", "", regex=False
    )
    data["form_normalized"] = data["form_normalized"].str.strip()
    data["form_normalized"] = data["form_normalized"].str.lower()

    data.dropna(subset=["form_normalized"], inplace=True)
    data = data[data["form_normalized"] != ""]

    print("Step 2: Establishing the base form (infinitive) for each verb paradigm...")

    # Filter for infinitives (Principal Part 1) - using numeric principal_part if available, else string "1"
    if "principal_part" in data.columns:
        infinitives = data[data["principal_part"] == 1].copy()
    else:
        infinitives = data[data.get("principal_part", "") == "1"].copy()

    # Most common form per lemma_id, corpus, variety
    base_forms = infinitives.groupby(["lemma_id", "corpus", "variety"])[
        "form_normalized"
    ].agg(lambda x: Counter(x).most_common(1)[0][0])
    base_form_dict = base_forms.to_dict()
    data["base_form"] = data.set_index(["lemma_id", "corpus", "variety"]).index.map(
        base_form_dict.get
    )
    data.dropna(subset=["base_form"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    print("Step 3: Developing and applying stem extraction logic...")

    # Apply stem extraction
    data["stem"] = data.apply(
        lambda r: strip_common_affixes(
            r["form_normalized"],
            r.get("lemma"),
            r.get("base_lemma"),
            r.get("corpus", "MHG"),
        ),
        axis=1,
    )
    data["base_stem"] = data.apply(
        lambda r: strip_common_affixes(
            r["base_form"], r.get("lemma"), r.get("base_lemma"), r.get("corpus", "MHG")
        ),
        axis=1,
    )

    print(
        "Step 4: Coding vowel and consonant alternations... (this can take time for large datasets)"
    )

    # Compute alternations row-wise
    alternations = data.progress_apply(
        compute_row_alternations,
        axis=1,
    )
    data = pd.concat([data, alternations], axis=1)
    data = data[
        data["vowel_alternation"].isin(["yes", "no"])
        & data["cons_alternation"].isin(["yes", "no"])
    ]
    data.reset_index(drop=True, inplace=True)

    # Save
    print(f"Saving coded output to: {output_csv}")
    data.to_csv(output_csv, index=False, encoding="utf-8")
    print("Done.")


# ---------------------------
# CLI
# ---------------------------


def cli():
    parser = argparse.ArgumentParser(
        description="Compute vowel and consonant alternations and write _coded CSV."
    )
    parser.add_argument("--input_csv", help="Path to the input CSV file")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional output path. By default inserts '_coded' before extension.",
    )
    args = parser.parse_args()
    process_file(args.input_csv, args.output)


if __name__ == "__main__":
    cli()
