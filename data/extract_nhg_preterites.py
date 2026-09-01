#!/usr/bin/env python3
"""
Modern German preterites from UniMorph (deu).

Source
------
UniMorph, German dataset.
  Repository : https://github.com/unimorph/deu
  Commit     : d226d2112d3490d8f04ece10d4538123d4297a39 (2024-07-24)
  File       : https://raw.githubusercontent.com/unimorph/deu/
               d226d2112d3490d8f04ece10d4538123d4297a39/deu
  SHA-256    : 8bc98487e3caf86188e2c072ebf2dc734f2ca35afef016bb399eb5a43e69d7c0
  Size       : 18,932,560 bytes, 519,143 inflectional entries
  Upstream   : the deu dataset is derived from English Wiktionary (repository
               README, "Source: `deu`: English Wiktionary").

Licence
-------
Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0).
Stated in the repository README at the pinned commit:
  https://github.com/unimorph/deu/blob/
  d226d2112d3490d8f04ece10d4538123d4297a39/README.md#license
  -> https://creativecommons.org/licenses/by-sa/3.0/
The repository carries no LICENSE file; the README section above is the licence
statement this project relies on.

Citation
--------
  Batsuren et al. (2022). UniMorph 4.0: Universal Morphology. LREC 2022.
  Kirov et al. (2018). UniMorph 2.0: Universal Morphology. LREC 2018.
  Sylak-Glassman (2016). The Universal Morphological Feature Schema.

Reproducibility
---------------
The download is pinned to the commit SHA above, not to `master`, and the cached
file is verified against the recorded SHA-256 on every run. A cache that does
not match is refused rather than used, so the provenance string written into
data/lemmas/nhg_targets.csv always describes the bytes that were actually read.
Pass --refresh to re-download.

What is extracted
-----------------
Tag `V;IND;SG;3;PST`: verb, indicative, singular, 3rd person, past. The bare
form, no pronoun and no whitespace padding: `litt`, `fand`, `verlor`, `schnitt`.

Variant policy
--------------
UniMorph lists more than one preterite for about 105 of the 6,659 verbs it
covers. Most are separable verbs this project never looks at; 17 are reachable
from data/lemmas/nhg_targets.csv and 8 of those are in the model. There is no
signal inside the source that ranks the variants: no frequency, no register
label, no ordering guarantee. The previous rule here was "take whichever form
does not end in -te". That is a string test standing in for a linguistic claim,
and it picked `muhl` for *mahlen* and `schrak` for *schrecken*, neither of which
is the form present-day German has settled on.

So the choice is made per lemma in VARIANT_POLICY below, with a reason, under
one criterion: the target of this study is the root the verb reached in
present-day standard German, so keep the strong preterite where it is still
current, and take the weak one where the strong form has dropped out or belongs
to a different verb.

Two guards keep this from becoming a back door for invented forms:

  * every chosen form is asserted to be present in the source's own variant list
    for that lemma, and the script fails loudly if it is not;
  * every variant the source offers is carried through to the output, so
    analysis/marking_type_summary.py can re-run the whole coding under the
    rejected variant and show what the choice is worth.

A multi-variant lemma with no policy entry is written out as `unresolved` with
an empty preterite rather than silently resolved by file order. That is only an
error if the table actually needs it, so build_nhg_targets.py is what raises,
naming the lemma. Verbs with a single UniMorph preterite are untouched by any of
this.

Output
------
  data/lemmas/unimorph_deu.tsv      raw cache, byte-identical to the pinned URL
  data/lemmas/unimorph_mapping.json {meta, forms}
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

UNIMORPH_COMMIT = "d226d2112d3490d8f04ece10d4538123d4297a39"
UNIMORPH_URL = f"https://raw.githubusercontent.com/unimorph/deu/{UNIMORPH_COMMIT}/deu"
UNIMORPH_SHA256 = "8bc98487e3caf86188e2c072ebf2dc734f2ca35afef016bb399eb5a43e69d7c0"
UNIMORPH_LICENCE = "CC BY-SA 3.0"

CACHE_PATH = Path("data/lemmas/unimorph_deu.tsv")
OUTPUT_MAPPING = Path("data/lemmas/unimorph_mapping.json")
SOURCE_TAG = f"unimorph-deu-{UNIMORPH_COMMIT[:7]}"

TARGET_TAG = "V;IND;SG;3;PST"

# lemma -> (chosen form, reason). Every chosen form must appear in the source's
# own list for that lemma; extract_preterites() enforces that. Verbs marked
# "in model" carry weight in the fitted data, the rest are listed so the whole
# column is deterministic rather than order-dependent.
VARIANT_POLICY: Dict[str, tuple] = {
    # --- in the model -------------------------------------------------------
    "hauen": ("hieb", "strong form still current for the 'hew, strike' sense of MHG houwen; haute is colloquial"),
    "hängen": ("hing", "hing is the intransitive reflex of MHG hâhen; hängte belongs to the transitive causative (MHG hengen)"),
    "mahlen": ("mahlte", "muhl has dropped out of present-day German; mahlte is the settled form"),
    "schrecken": ("schreckte", "the simplex schrecken is weak today; the strong schrak survives only in the prefixed erschrak, so it is not the endpoint of this headword. Flip to 'schrak' to treat the prefixed form as the continuation"),
    "schwimmen": ("schwamm", "schwomm has dropped out of the standard language"),
    "sieden": ("sott", "the strong sott is still current, if technical and literary"),
    "triefen": ("troff", "the strong troff is still current, if literary; triefte is the commoner variant, which makes this the weakest call in this table"),
    "wachsen": ("wuchs", "wachste is the weak homograph wachsen 'apply wax', a different verb from MHG wahsen 'grow'"),
    # --- not in the model, listed for determinism ----------------------------
    "backen": ("backte", "buk has dropped out of ordinary standard use; backte is the settled form"),
    "flechten": ("flocht", "strong form is the standard one; flechtete is rare"),
    "saugen": ("sog", "strong form still current"),
    "schaffen": ("schuf", "schaffte belongs to the weak homograph schaffen 'manage, work'; MHG schaffen 'create' continues as schuf"),
    "scheren": ("schor", "strong form still current"),
    "senden": ("sandte", "Rückumlauf form is the commoner standard variant"),
    "weben": ("wob", "strong form still current, if elevated"),
    "wenden": ("wandte", "Rückumlauf form is the commoner standard variant"),
    "werden": ("wurde", "ward is archaic; wurde is the modern form"),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_unimorph(dest: Path = CACHE_PATH, refresh: bool = False) -> Path:
    """
    Return a local copy of the pinned UniMorph file, downloading it once.

    The checksum is verified whether the file was just fetched or came from the
    cache. A mismatch raises: a silently wrong cache would put a provenance
    string on rows that did not come from that revision.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        found = sha256_of(dest)
        if found != UNIMORPH_SHA256:
            raise SystemExit(
                f"Cached {dest} does not match the pinned revision.\n"
                f"  expected sha256 {UNIMORPH_SHA256}\n"
                f"  found    sha256 {found}\n"
                f"Delete the file, or re-run with --refresh, to fetch {UNIMORPH_COMMIT[:7]} again."
            )
        print(f"Using cached UniMorph {UNIMORPH_COMMIT[:7]}: {dest} ({dest.stat().st_size:,} bytes, sha256 ok)")
        return dest

    print(f"Downloading UniMorph {UNIMORPH_COMMIT[:7]} from {UNIMORPH_URL}")
    req = urllib.request.Request(UNIMORPH_URL, headers={"User-Agent": "prop_hazards_verbs"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read()

    found = hashlib.sha256(content).hexdigest()
    if found != UNIMORPH_SHA256:
        raise SystemExit(
            f"Downloaded file does not match the pinned revision.\n"
            f"  expected sha256 {UNIMORPH_SHA256}\n"
            f"  found    sha256 {found}\n"
            f"The URL is pinned to a commit SHA, so this should not happen. Do not use the result."
        )

    dest.write_bytes(content)
    print(f"Cached {len(content):,} bytes to {dest} (sha256 ok)")
    return dest


def read_variants(tsv_path: Path = CACHE_PATH) -> Dict[str, List[str]]:
    """lemma -> the 3sg preterite indicative forms the source lists, in file order."""
    variants: Dict[str, List[str]] = {}
    with open(tsv_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            lemma, form, tag = (p.strip() for p in parts)
            if tag != TARGET_TAG or not form:
                continue
            seen = variants.setdefault(lemma, [])
            if form not in seen:
                seen.append(form)
    return variants


def extract_preterites(tsv_path: Path = CACHE_PATH) -> Dict[str, dict]:
    """
    lemma -> {"preterite", "variants", "selection", "note"}.

    `selection` is "single" when the source offers one form, "policy" when
    VARIANT_POLICY resolved a choice, and "unresolved" when the source offers
    several and no policy entry covers the lemma. An unresolved entry carries an
    empty preterite: file order is not a linguistic argument, so it is not used
    as a tie-break. build_nhg_targets.py raises if the table needs one.
    """
    variants = read_variants(tsv_path)

    forms: Dict[str, dict] = {}
    for lemma, options in variants.items():
        if len(options) == 1:
            forms[lemma] = {
                "preterite": options[0],
                "variants": options,
                "selection": "single",
                "note": "",
            }
            continue

        policy = VARIANT_POLICY.get(lemma)
        if policy is None:
            forms[lemma] = {
                "preterite": "",
                "variants": options,
                "selection": "unresolved",
                "note": "several UniMorph preterites and no VARIANT_POLICY entry",
            }
            continue

        chosen, reason = policy
        if chosen not in options:
            raise SystemExit(
                f"VARIANT_POLICY picks {chosen!r} for {lemma!r}, but UniMorph "
                f"{UNIMORPH_COMMIT[:7]} lists {options}. The policy may only choose "
                f"among forms the source supplies."
            )
        forms[lemma] = {
            "preterite": chosen,
            "variants": options,
            "selection": "policy",
            "note": reason,
        }

    for lemma in VARIANT_POLICY:
        if lemma not in variants:
            print(f"  note: VARIANT_POLICY mentions {lemma!r}, which this revision does not list", file=sys.stderr)

    return forms


def build_mapping(tsv_path: Path = CACHE_PATH) -> dict:
    forms = extract_preterites(tsv_path)
    return {
        "meta": {
            "source": "UniMorph deu",
            "source_tag": SOURCE_TAG,
            "commit": UNIMORPH_COMMIT,
            "url": UNIMORPH_URL,
            "sha256": UNIMORPH_SHA256,
            "tag": TARGET_TAG,
            "licence": UNIMORPH_LICENCE,
        },
        "forms": forms,
    }


def load_mapping(path: Path = OUTPUT_MAPPING, tsv_path: Path = CACHE_PATH) -> Dict[str, dict]:
    """Read the cached mapping, building it first if it is absent."""
    if not path.exists():
        if not tsv_path.exists():
            download_unimorph(tsv_path)
        mapping = build_mapping(tsv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(mapping, handle, ensure_ascii=False, indent=2)
        return mapping["forms"]

    with open(path, encoding="utf-8") as handle:
        mapping = json.load(handle)
    return mapping["forms"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--refresh", action="store_true", help="re-download even if the cache is present and valid")
    args = parser.parse_args()

    tsv_path = download_unimorph(refresh=args.refresh)
    mapping = build_mapping(tsv_path)

    OUTPUT_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MAPPING, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)

    forms = mapping["forms"]
    by_policy = sorted(l for l, e in forms.items() if e["selection"] == "policy")
    unresolved = sorted(l for l, e in forms.items() if e["selection"] == "unresolved")
    print(f"Extracted {len(forms):,} verb preterites into {OUTPUT_MAPPING}")
    print(f"  single form in the source  : {sum(1 for e in forms.values() if e['selection'] == 'single'):,}")
    print(f"  resolved by VARIANT_POLICY : {len(by_policy)}")
    for lemma in by_policy:
        entry = forms[lemma]
        rejected = [v for v in entry["variants"] if v != entry["preterite"]]
        print(f"    {lemma:<12} -> {entry['preterite']:<10} (over {', '.join(rejected)})")
    print(f"  left unresolved            : {len(unresolved)} (mostly separable verbs; an error only if the table needs one)")


if __name__ == "__main__":
    main()
