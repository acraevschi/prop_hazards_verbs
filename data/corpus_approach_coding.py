import pandas as pd
import os
import csv
import re
from tqdm import tqdm
import unicodedata

# ------------------------------------------------------------------------------
# 1. CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------------------

# Vowels (MHG + ENHG variants)
VOWELS = set("aeiouäöüyoͮuͦeͦuͤoͤeͤaͤu͗âāáêēëèéᵉîīíyÿýôōûūæœ")

# Normalization Maps
VOWEL_NORM_DICT = {
    "oͮ": "o",
    "uͦ": "u",
    "eͦ": "o",
    "vͦ": "ü",
    "uͤ": "u",
    "oͤ": "o",
    "eͤ": "e",
    "aͤ": "a",
    "u͗": "u",
    "v͗": "ü",
    "w͗": "w",
    "v̈": "ü",
    "â": "a",
    "ā": "a",
    "á": "a",
    "ê": "e",
    "ē": "e",
    "ë": "e",
    "è": "e",
    "é": "e",
    "ᵉ": "e",
    "î": "i",
    "ī": "i",
    "í": "i",
    "y": "i",
    "ÿ": "i",
    "ý": "i",
    "ô": "o",
    "ō": "o",
    "û": "u",
    "ū": "u",
    "æ": "ä",
    "œ": "ö",
}

CONS_NORM_DICT = {
    # Macrons/Titulus (scribal abbreviations for m/n or gemination)
    "n̄": "n",
    "m̄": "m",
    "h̄": "h",
    "t̄": "t",
    "ḡ": "g",
    "̄": "n",  # Standalone combining macron often implies 'n' or 'm'
    # Circumflexes/breves on consonants
    "m̂": "m",
    "n̂": "n",
    "ĝ": "g",
    "ĉ": "c",
    "ŝ": "s",
    # Historic Ligatures & Variants
    "ß": "s",
    "ſ": "s",
    "ʃ": "s",
    "ʒ": "s",
    "z": "z",
    "ʦ": "z",
    "ꝛ": "r",
    "ꝝ": "r",  # R rotunda variants
    # Superscripts (often ignored or mapped)
    "ˢ": "",
    "ᵐ": "m",
    "ⁿ": "n",
    "ʳ": "r",
}

DIGRAPH_IPA = {
    "sch": "ʃ",
    "ch": "χ",
    "ck": "k",
    "ph": "f",
    "pf": "f",
    "th": "t",
    "tz": "ʦ",
    "ts": "ʦ",
    "sz": "s",
    "cz": "z",
    "vb": "üb",
    "vn": "un",
    "nc": "nk",
    "czt": "ʦ",
    "nph": "mf",
    "cr": "kr",
    "enʦ": "s",
    "vˢ": "ver",
    "qu": "kw",
}

# Equivalence Sets for Consonants (Spelling/Devoicing variants)
# If a change is found within these sets, it is NOT Grammatischer Wechsel
EQUIV_SETS = [
    {"v", "f", "u"},  # u/v/f interchange
    {"p", "b"},  # Devoicing
    # {"t", "d"},  # Devoicing
    {"h", "χ"},
    {"k", "g", "c", "q", "ng", "nk"},  # Devoicing / Spelling
    {"s", "z", "ʦ", "ʃ", "ʒ", "ss", "ß"},  # Sibilants
]

# The voicing pairs that Auslautverhärtung can produce in a word-final coda,
# and that are therefore ambiguous between devoicing and Verner's Law when the
# past singular is one of the two cells compared. EQUIV_SETS already treats
# p ~ b and k ~ g as spelling variants, so they never reach a GW test; the
# dentals are held apart there on purpose, because d ~ t is the Class I
# grammatischer Wechsel, and they are what this set has to name.
DEVOICING_PAIRS = frozenset({frozenset({"t", "d"})})

PREFIXES = {
    "MHG": [
        "ge",
        "be",
        "er",
        "ver",
        "zer",
        "ent",
        "emp",
        "enʦ",
        "miss",
        "zuo",
        "unter",
        "über",
    ],
    "ENHG": [
        "ge",
        "be",
        "er",
        "ver",
        "zer",
        "ent",
        "emp",
        "miss",
        "zuo",
        "unter",
        "über",
    ],
}

SUFFIXES = [
    "ende",
    "ente",
    "nden",
    "nten",
    "est",
    "ent",
    "en",
    "et",
    "es",
    "st",
    "t",
    "e",
    "n",
    "d",
]

VALID_ONSETS = {
    "bl",
    "br",
    "cl",
    "cr",
    "dr",
    "fl",
    "fr",
    "gl",
    "gn",
    "gr",
    "kl",
    "kn",
    "kr",
    "pf",
    "ph",
    "pl",
    "pr",
    "qu",
    "sch",
    "sl",
    "sm",
    "sn",
    "sp",
    "st",
    "sw",
    "tr",
    "tw",
    "vl",
    "vr",
    "wr",
    "zw",
}

# Clusters that can only ever be a syllable coda. Seeing one as the onset of a
# candidate stem means the strip cut into the root: ge- off gelten leaves lten,
# be- off bergen leaves rgen.
CODA_CLUSTERS = (
    "lt",
    "rg",
    "rc",
    "rb",
    "lz",
    "lm",
    "rm",
    "rn",
    "ld",
    "nd",
    "ng",
    "nk",
    "nt",
)

# Subparadigms whose forms carry a consonantal inflectional ending.
#
# The past indicative singular is endingless in MHG: gap, nam, was, wan, gewan.
# A final -n there is part of the root, not an infinitive or plural marker.
# Stripping it anyway is what turned gewan into ge- plus wan with the coda w,
# which handed ge-winnen a consonant alternation the verb has never had. Every
# other slot does take an ending, so -n on a vowel-final root is removed there
# (gan, stan, schrien).
SLOTS_WITH_ENDING = {"Pres", "PastPl", "Ppl"}


def load_sound_changes(filepath="data/vowel_changes.csv"):
    """
    Loads vowel changes into a dictionary:
    {(Variety, MHG_Vowel): set([ENHG_Vowel_Options])}
    """
    changes = {}
    if not os.path.exists(filepath):
        print(
            f"Warning: {filepath} not found. Sound change filtering will be disabled."
        )
        return changes

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize headers just in case
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row in reader:
            mhg = row["MHG"].strip()
            enhg = row["ENHG"].strip()
            variety = row["Variety"].strip()

            # Create a key for the specific variety
            key = (variety, mhg)
            if key not in changes:
                changes[key] = set()
            changes[key].add(enhg)

            # Also add a general fallback if variety might be missing or slight mismatch
            # (Optional, but safe for robustness)
            gen_key = ("General", mhg)
            if gen_key not in changes:
                changes[gen_key] = set()
            changes[gen_key].add(enhg)

    return changes


# ------------------------------------------------------------------------------
# 2. CORE MORPHOLOGY FUNCTIONS
# ------------------------------------------------------------------------------


def standardize_infl(val):
    """
    Normalizes inflection labels to the three critical slots.

    This is the fallback for rows that carry no principal_part. The authority is
    map_category in data/normalize_data.py, and this function must agree with
    it. The four Germanic principal parts are:

        PP1  infinitive and present
        PP2  past indicative singular, 1st and 3rd person
        PP3  past indicative singular 2nd person, past plural, and the whole
             past subjunctive
        PP4  past participle

    PP3 groups those cells because they share one stem. The past subjunctive is
    built on the past plural stem with umlaut: hulfen -> hülfe, zugen -> züge.
    Reading mood therefore has to come before number: a past subjunctive
    singular belongs with the plural, not with half -> PastSg. Without the mood
    test, hülfe looks like a singular that has taken the plural vowel, which is
    the leveling event this study counts.
    """
    if pd.isna(val):
        return "Pres"
    val = str(val).lower()
    if "participle" in val or "ppl" in val:
        return "Ppl"
    if "past" in val or "prät" in val or "pret" in val:
        if "subj" in val or "konj" in val:  # past subjunctive is built on the plural stem
            return "PastPl"
        if "pl" in val or "2" in val:  # 2nd person patterns with the plural in MHG
            return "PastPl"
        return "PastSg"
    return "Pres"


# principal_part -> subparadigm, as map_category in data/normalize_data.py
# numbers them.
PRINCIPAL_PART_TO_INFL = {1: "Pres", 2: "PastSg", 3: "PastPl", 4: "Ppl"}


def are_cons_equivalent(c1, c2, protected=None):
    """
    Returns True if c1 and c2 are phonologically or orthographically equivalent
    (i.e., NOT Grammatischer Wechsel).

    `protected` works as in are_vowels_equivalent: a contrast that the paradigm
    carries in its own pre-1200 baseline is never equated away.
    """
    if pd.isna(c1) or pd.isna(c2):
        return False

    c1, c2 = str(c1).lower(), str(c2).lower()
    c1 = c1[-1] if len(c1) > 1 else c1
    c2 = c2[-1] if len(c2) > 1 else c2
    if c1 == c2:
        return True
    if protected and frozenset((c1, c2)) in protected:
        return False
    for s in EQUIV_SETS:
        if c1 in s and c2 in s:
            return True
    return False


def are_vowels_equivalent(v1, v2, variety, sc_dict, protected=None):
    """
    Returns True if v1 == v2 OR if the transition v1->v2 (or v2->v1)
    is a regular sound change in the given variety.

    `protected` holds the contrasts that the paradigm carries in its own
    pre-1200 baseline, as a set of frozensets. A regular sound change must not
    explain away a contrast that the baseline already established.

    This guard is necessary because root extraction discards vowel length. MHG
    long i and short i both become "i", so the diphthongization rule i -> ei
    also licenses short i against ei. Short i against ei is the Class I ablaut
    alternation, so without this guard the filter erases the very alternation
    that the study measures. The same happens to Class II o ~ u in Central
    German through the rule u -> o.
    """
    if pd.isna(v1) or pd.isna(v2):
        return False

    v1, v2 = str(v1), str(v2)

    # 1. Strict Identity
    if v1 == v2:
        return True

    # 2. Baseline contrast. The paradigm distinguishes these two roots, so no
    # sound change may merge them. Identity above still wins, because a root
    # cannot contrast with itself.
    if protected and frozenset((v1, v2)) in protected:
        return False

    # 3. Sound Change Lookup
    # We check both directions because we might be comparing Anchor(MHG)->Obs(ENHG)
    # or Obs(ENHG)->Anchor(MHG).

    # Check v1 (MHG) -> v2 (ENHG)
    valid_outcomes = sc_dict.get((variety, v1), set())
    if v2 in valid_outcomes:
        return True

    # Check v2 (MHG) -> v1 (ENHG)
    valid_outcomes_rev = sc_dict.get((variety, v2), set())
    if v1 in valid_outcomes_rev:
        return True

    return False


def clean_form(form):
    """
    Robust cleaning that strips combining diacritics from medieval texts.

    Known limitation: an orthographic u that spells /v/ before a vowel is left
    as a vowel. ReF writes gevallen as geuallen and bevolhen as beuolhen, so the
    prefix is not recognised and the prefix vowel joins the root nucleus
    (geuallen -> ('eua', 'l'), where the root is ('a', 'l')). Deciding u from v
    needs the surrounding graphotactics and often the lemma, and no rule
    separated geuallen from a genuine u-initial stem without new errors. Such
    rows match neither anchor nor target and are coded NA, so the cost is lost
    data rather than false events: 40 of 49,616 coded rows, all in ReF. See
    "Known Limitations of the Coding Pipeline" in README.md.
    """
    if pd.isna(form):
        return ""
    f = str(form).lower().strip()

    # 1. Remove editorial brackets first
    f = re.sub(r"\[.*?\]", "", f)

    # 2. Specific Dictionary Normalization (Hand-picked replacements)
    for k, v in VOWEL_NORM_DICT.items():
        f = f.replace(k, v)
    for k, v in CONS_NORM_DICT.items():
        f = f.replace(k, v)

    # 3. "Nuclear Option": Strip remaining combining diacritics
    # This removes the floating squiggles seen in your screenshot (like the leftover macrons)
    # NFD separates base chars from accents (e.g., 'ñ' -> 'n' + '~')
    # We then keep only non-combining characters.
    f_norm = unicodedata.normalize("NFD", f)
    f = "".join([c for c in f_norm if not unicodedata.combining(c)])

    # 4. qu -> kw, before anything reads the string for vowels.
    # DIGRAPH_IPA carries this mapping too, but it is applied to the nucleus
    # after extraction, which is too late: the vowel regex has already taken the
    # u of qu as part of the nucleus. That is why quam, the past singular of
    # komen and one of the most frequent past forms in MHG, came out with the
    # nucleus "ua" instead of "a".
    f = f.replace("qu", "kw")

    # 5. Final filter for valid alphabet (Optional, but safe)
    # Keep standard German alphabet + common IPA chars you use
    # Note: \u00DF is ß
    f = re.sub(r"[^a-zäöüßſ\u00E6\u0153\u0283\u02A6\u03C7]", "", f)

    return f


def _lemma_key(value):
    """lemma_id as a string, tolerating the float form pandas hands back."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text[:-2] if text.endswith(".0") else text


def _onset_of(stem):
    """The consonants before the first vowel. Empty for a vowel-initial root."""
    match = re.match(f"^([^{''.join(VOWELS)}]*)", stem)
    return match.group(1) if match else ""


def _nucleus_and_coda(stem, std_infl=None):
    """
    Split a prefix-free stem into its first vowel run and the consonants that
    follow it, with inflectional endings removed.

    std_infl decides the one ambiguous case: a coda consisting of nothing but
    -n. In the present, the past plural and the participle that -n is an ending
    and the root is vowel-final (gan, stan, schrien). In the past singular there
    is no ending, so the -n is the root's own coda (wan, gewan, began). See
    SLOTS_WITH_ENDING. A missing std_infl is treated as ending-bearing, which is
    the behaviour every caller had before the argument existed.

    Returns (None, None) when the stem has no vowel.
    """
    match = re.search(f"([{''.join(VOWELS)}]+)", stem)
    if not match:
        return None, None

    nucleus = match.group(1)
    post_nucleus = stem[match.end() :]
    coda_match = re.match(f"^([^{''.join(VOWELS)}]+)", post_nucleus)
    coda = coda_match.group(1) if coda_match else ""

    takes_ending = std_infl is None or std_infl in SLOTS_WITH_ENDING

    for s in sorted(SUFFIXES, key=len, reverse=True):
        if not coda.endswith(s):
            continue
        if len(coda) > len(s):
            coda = coda[: -len(s)]
            break
        # Deleting the coda outright is only ever right for the -n of a
        # vowel-final root, and only in a slot that takes an ending.
        if s in ("n", "en") and takes_ending:
            coda = ""
            break
        # Otherwise protect it (the t of gilt, the p of gap).

    return nucleus, coda


def build_lemma_index(df):
    """
    Read each lemma_id's morphology off the lemma strings the corpus carries.

    ReM writes a prefixed lemma with a hyphen - ge-winnen, ver-lièsen,
    über-winten, ent-vâhen - and a separable particle after a slash
    (slahen/abe>+). That hyphen is the corpus telling us where the root starts,
    and it is the only signal that separates ge-winnen, where ge- is a prefix,
    from gëben, where it is not. No phonotactic rule can make that distinction,
    because ge+ben and ge+wan are the same shape.

    ReF lemmas are modern infinitives with no hyphens (verlieren, behalten), so
    their prefixes are unmarked and reading them would give a wrong root onset.
    They are ignored. ENHG rows inherit the analysis through lemma_id, which is
    the key the rest of the pipeline already joins on. 228 of the 292 lemma_ids
    get an entry this way, covering 133 of the 197 that appear in ReF; the rest
    fall back to the phonotactic guards in extract_root_structure.

    Returns {lemma_id: {"prefixes": set, "onsets": set}}.
    """
    index = {}
    if df is None or not hasattr(df, "columns"):
        return index
    if "lemma" not in df.columns or "lemma_id" not in df.columns:
        return index

    rows = df[["lemma_id", "lemma"]]
    if "corpus" in df.columns:
        rows = df.loc[df["corpus"] == "MHG", ["lemma_id", "lemma"]]
    rows = rows.dropna()

    for lemma_id, group in rows.groupby("lemma_id"):
        key = _lemma_key(lemma_id)
        if key is None:
            continue
        prefixes, onsets = set(), set()
        for lemma in group["lemma"].unique():
            base = str(lemma).split("/")[0]  # drop the separable particle
            parts = base.split("-")
            root = clean_form(parts[-1])
            if not root:
                continue
            for part in parts[:-1]:
                prefix = clean_form(part)
                if prefix:
                    prefixes.add(prefix)
                    prefixes.update(PREFIX_SPELLING_VARIANTS.get(prefix, ()))
            onsets.add(_onset_of(root))
        if onsets:
            index[key] = {"prefixes": prefixes, "onsets": onsets}

    return index


# Onset spellings that the regular MHG -> ENHG correspondences make equivalent.
# build_lemma_index reads its roots off ReM lemma strings, so a ReF form has to
# be recognised across the sound and spelling changes that separate the two
# corpora, or every ENHG participle of vâhen, slahen and binden looks like a
# different verb and keeps its prefix.
ONSET_CLUSTER_FOLD = (("sch", "s"), ("ʃ", "s"))

# Onsets that a ReF scribe writes for a single sound. Unlike ONSET_CLUSTER_FOLD
# these replace the whole onset, because ph and pf ARE the onset in enphangen
# and enpfangen, where MHG wrote v: ent-vâhen -> empfangen.
ONSET_WHOLE_FOLD = {"ph": "f", "pf": "f", "ff": "f", "v": "f"}

# ReF spellings of the MHG prefixes that build_lemma_index reads off ReM lemma
# strings. ent- assimilates to the following labial and its dental is written
# t, p or nothing at all: ent-vâhen surfaces as entphangen, enpfangen,
# enphangen, empfangen and intfangen. Without these the prefix goes unstripped
# and the root is lost. Keyed on the MHG prefix so that only lemmas whose own
# lemma string carries it are affected.
PREFIX_SPELLING_VARIANTS = {
    "ent": ("ent", "entp", "entph", "entpf", "emp", "enp", "enph", "enpf", "en", "int"),
}
ONSET_LETTER_FOLD = str.maketrans(
    {
        "v": "f",  # vâhen ~ fangen, vallen ~ fallen
        "u": "f",  # u and v are allographs
        "p": "b",  # binden ~ gepunden, Upper German
        "t": "d",  # and the same for the dentals, which also folds tw ~ dw
        "c": "k",
        "q": "k",
        "ʦ": "z",
        "ß": "s",
        "χ": "h",
    }
)


def _fold_onset(onset):
    """Reduce an onset to a shape that survives ReM/ReF spelling differences."""
    if onset in ONSET_WHOLE_FOLD:
        return ONSET_WHOLE_FOLD[onset].translate(ONSET_LETTER_FOLD)
    for long_form, short_form in ONSET_CLUSTER_FOLD:
        # slahen ~ schlagen, snîden ~ schneiden, swimmen ~ schwimmen. Only when
        # something follows, so that sch- before a vowel (schehen) is left alone.
        if onset.startswith(long_form) and len(onset) > len(long_form):
            onset = short_form + onset[len(long_form) :]
            break
    return onset.translate(ONSET_LETTER_FOLD)


def _onset_is_compatible(onset, root_onsets, prefix):
    """
    Does a candidate stem begin the way this lemma's root begins?

    Compared after _fold_onset on both sides. Exact match is the normal case.
    Two relaxations are allowed:

    - One onset extending the other, when both are non-empty. This covers a past
      stem whose onset is spelled longer than the infinitive's (komen against
      kwam, from quam). The both-non-empty condition is what keeps be- on beiz:
      the candidate iz has an empty onset, and an empty string is a prefix of
      everything, so without it every vowel-initial remainder would be accepted.
    - The prefix's own final consonant completing the onset. German degeminates
      at the prefix boundary, so ent- plus trinnen is written entran, and the
      candidate ran has to be read against the root onset tr.
    """
    if not root_onsets:
        return True
    if onset in root_onsets:
        return True

    folded = _fold_onset(onset)
    folded_roots = {_fold_onset(r) for r in root_onsets}

    if folded in folded_roots:
        return True
    if folded:
        for root_onset in folded_roots:
            if root_onset and (
                folded.startswith(root_onset) or root_onset.startswith(folded)
            ):
                return True
    if prefix and _fold_onset(prefix[-1] + onset) in folded_roots:
        return True
    return False


def extract_root_structure(
    df, form, corpus="MHG", lemma_id=None, std_infl=None, lemma_index=None
):
    """
    Split a surface form into root nucleus and root coda.

    Prefix stripping is decided by the lemma whenever build_lemma_index has an
    entry for lemma_id: a prefix comes off only if what is left begins the way
    that verb's root begins. This is what keeps ge- on gëben while taking it off
    ge-winnen, and it is why geaz parses as az even though the root is
    vowel-initial, which no phonotactic guard can allow without also letting
    be- eat the b of beiz.

    Without an entry the function falls back to phonotactic guards. They are
    weaker by construction - they reject every vowel-initial remainder, which is
    wrong for ezzen - so they are a floor, not the intended path.

    std_infl is passed through to _nucleus_and_coda, where it settles whether a
    final -n is an ending or root material.
    """
    f = clean_form(form)
    if not f:
        return pd.NA, pd.NA

    entry = (lemma_index or {}).get(_lemma_key(lemma_id))
    root_onsets = entry["onsets"] if entry else None

    # --- 1. PREFIX STRIPPING ---
    # clean_form the prefixes too. The list holds "über", the forms have had
    # their diacritics stripped to "uber", and a raw comparison never matched.
    prefix_list = [clean_form(p) for p in PREFIXES.get(corpus, PREFIXES["MHG"])]
    if entry:
        # The lemma may name prefixes the generic list does not carry:
        # en-bîzen, umbe-, durh-, misse-, vol-.
        prefix_list.extend(entry["prefixes"])
    prefix_list = sorted({p for p in prefix_list if p}, key=len, reverse=True)

    clean_stem = f

    # Safety loop limit to prevent infinite loops (though unlikely)
    for _ in range(3):
        match_found = False
        for p in prefix_list:
            if not clean_stem.startswith(p):
                continue
            candidate = clean_stem[len(p) :]

            # A strip that removes the nucleus is never right.
            if not any(v in candidate for v in VOWELS):
                continue

            onset = _onset_of(candidate)

            # No root starts with a geminate (ge- off gessen) or with a cluster
            # that can only be a coda (ge- off gelten, be- off bergen).
            if len(onset) > 1 and (
                onset[0] == onset[1] or onset.startswith(CODA_CLUSTERS)
            ):
                continue

            if root_onsets is not None:
                if not _onset_is_compatible(onset, root_onsets, p):
                    continue
            else:
                # No lemma analysis. Assume a vowel-initial remainder means the
                # strip cut into the root, which is right for beiz and wrong for
                # the handful of vowel-initial roots the lemma path covers.
                if not onset:
                    continue
                # And refuse a strip that would leave the root with no coda at
                # all while the unstripped stem still has one: ge- off gëben
                # leaves ben, whose only consonant after the nucleus is the -n
                # of the infinitive.
                if (
                    not _nucleus_and_coda(candidate, std_infl)[1]
                    and _nucleus_and_coda(clean_stem, std_infl)[1]
                ):
                    continue

            clean_stem = candidate
            match_found = True
            break  # Restart loop to check for next prefix (e.g. vor-ge-)

        if not match_found:
            break

    # --- 2. NUCLEUS AND CODA ---
    nucleus, final_coda = _nucleus_and_coda(clean_stem, std_infl)
    if nucleus is None:
        return pd.NA, pd.NA

    # --- 3. CLEANUP ---
    for k, v in DIGRAPH_IPA.items():
        final_coda = final_coda.replace(k, v)
        nucleus = nucleus.replace(k, v)

    final_coda = re.sub(r"(.)\1+", r"\1", final_coda)  # Degeminate

    return nucleus, final_coda


# ------------------------------------------------------------------------------
# 3. ANALYSIS PIPELINE
# ------------------------------------------------------------------------------


def step_1_preprocessing(df, lemma_index=None):
    print("--- Step 1: Preprocessing & Root Extraction (Lemma-Aware) ---")
    df = df.copy()

    df = df[~df["inflClass"].isin(["St|Sw|Unr", "Unr", "irr", "St|Unr", "prpr"])]

    df["date"] = pd.to_numeric(df["date"], errors="coerce")

    # The subparadigm comes from principal_part, which normalize_data.py already
    # assigns from mood, tense, number and person. Re-deriving it from the raw
    # label here duplicated that decision and disagreed with it on every past
    # subjunctive singular. standardize_infl stays as the fallback for rows that
    # reach this function without a principal_part.
    fallback = df["infl"].apply(standardize_infl)
    if "principal_part" in df.columns:
        df["std_infl"] = (
            pd.to_numeric(df["principal_part"], errors="coerce")
            .map(PRINCIPAL_PART_TO_INFL)
            .fillna(fallback)
        )
    else:
        df["std_infl"] = fallback

    if lemma_index is None:
        lemma_index = build_lemma_index(df)
    print(f"   Lemma morphology available for {len(lemma_index)} lemma_ids.")

    # We can no longer just map unique forms blindly, because 'geben' (Inf) and
    # 'geben' (some other context) might differ if we use lemma info.
    # std_infl is part of the key because extraction now depends on it: the
    # final -n of gewan is root material in the past singular and an ending
    # everywhere else, so the same string can parse two ways.

    context_cols = ["norm", "corpus", "lemma_id", "std_infl"]
    unique_contexts = df[context_cols].drop_duplicates()

    print(f"   Extracting roots for {len(unique_contexts)} unique form contexts...")

    results = []
    for _, row in tqdm(unique_contexts.iterrows(), total=len(unique_contexts)):
        # PASS THE LEMMA HERE
        v, c = extract_root_structure(
            df,
            row["norm"],
            row["corpus"],
            row["lemma_id"],
            std_infl=row["std_infl"],
            lemma_index=lemma_index,
        )
        results.append(
            {
                "norm": row["norm"],
                "corpus": row["corpus"],
                "lemma_id": row["lemma_id"],
                "std_infl": row["std_infl"],
                "extracted_vowel": v,
                "extracted_coda": c,
            }
        )

    # Map back
    res_df = pd.DataFrame(results)
    df = df.merge(res_df, on=context_cols, how="left")

    return df


def step_2_establish_baseline(df):
    """
    Establishes the 'Start State' (MHG Pre-1200).
    Calculates pairwise complexity (Ablaut/GW) between ALL three slots.
    """
    print("\n--- Step 2: Establishing Diachronic Baseline (Pre-1200) ---")

    # Filter for Baseline Candidates
    baseline_df = df[
        (df["date"] <= 1200) & (df["corpus"] == "MHG") & (df["extracted_vowel"].notna())
    ].copy()

    # Group by Lemma+Variety+Infl to get the mode
    anchors = (
        baseline_df.groupby(["lemma_id", "variety", "std_infl"])[
            ["extracted_vowel", "extracted_coda"]
        ]
        .agg(lambda x: pd.Series.mode(x)[0] if not x.mode().empty else pd.NA)
        .reset_index()
    )

    # Pivot to get one row per Lemma+Variety
    anchors_pivoted = anchors.pivot_table(
        index=["lemma_id", "variety"],
        columns="std_infl",
        values=["extracted_vowel", "extracted_coda"],
        aggfunc="first",
    )

    # Flatten MultiIndex columns
    anchors_pivoted.columns = [
        f"anchor_{x.split('_')[1]}_{y.lower()}" for x, y in anchors_pivoted.columns
    ]
    anchors_pivoted = anchors_pivoted.reset_index()

    # --- NEW LOGIC: Calculate Pairwise Distinctions ---

    def _coda_known(row, slot):
        """
        True when the paradigm actually attests a coda for this cell.

        A shape test that reads "the other cell agrees" must not be satisfied by
        a cell the baseline never resolved. check_alternation collapses "agrees"
        and "absent" into the same False, so the bipartite rule asks this
        separately.
        """
        value = row.get(f"anchor_coda_{slot}")
        return not pd.isna(value) and str(value).strip() != ""

    def _last_chars(row, slot1, slot2):
        """The frozenset of the two codas' final characters, as EQUIV_SETS compares them."""
        out = []
        for slot in (slot1, slot2):
            value = row.get(f"anchor_coda_{slot}")
            if pd.isna(value):
                return frozenset()
            text = str(value).strip().lower()
            if not text:
                return frozenset()
            out.append(text[-1])
        return frozenset(out)

    def check_alternation(row, slot1, slot2):
        """
        Returns (has_ablaut, has_gw) for two specific slots.
        """
        v1 = row.get(f"anchor_vowel_{slot1}")
        c1 = row.get(f"anchor_coda_{slot1}")
        v2 = row.get(f"anchor_vowel_{slot2}")
        c2 = row.get(f"anchor_coda_{slot2}")

        # 1. Null Check
        if pd.isna(v1) or pd.isna(v2) or pd.isna(c1) or pd.isna(c2):
            return False, False

        # Ablaut Check (Simple inequality)
        has_ablaut = v1 != v2

        # 2. GW Check (With Empty String Protection)
        # Ensure we are working with strings
        s1 = str(c1).strip()
        s2 = str(c2).strip()

        # CRITICAL FIX: If either consonant is empty, we DO NOT count it as GW.
        # This handles extraction errors or vowel-final roots safely.
        if not s1 or not s2:
            has_gw = False
        else:
            # Only compare if both exist
            has_gw = (s1 != s2) and (not are_cons_equivalent(s1, s2))

        return has_ablaut, has_gw

    # Apply calculations for all 3 pairs
    def analyze_paradigm(row):
        # 1. Pres vs PastSg
        ab_pres_sg, gw_pres_sg = check_alternation(row, "pres", "pastsg")
        # 2. Pres vs PastPl
        ab_pres_pl, gw_pres_pl = check_alternation(row, "pres", "pastpl")
        # 3. PastSg vs PastPl (The classic definition)
        ab_sg_pl, gw_sg_pl = check_alternation(row, "pastsg", "pastpl")

        # Global Bipartite Definition.
        #
        # The past indicative singular is the one endingless cell, so its coda
        # is word-final and subject to Auslautverhärtung. That makes every
        # consonant comparison involving it ambiguous between Verner's Law and
        # plain final devoicing, and the two have opposite paradigmatic shapes:
        #
        #   Verner        the past PLURAL is the odd cell out
        #                 wesen  ~ was  ~ wâren     s  ~ s ~ r
        #                 kiesen ~ kôs  ~ kurn      s  ~ s ~ r
        #                 ziehen ~ zôch ~ zugen     h  ~ χ ~ g
        #                 quëden ~ quat ~ quâden    t  ~ t ~ d
        #
        #   devoicing     the past SINGULAR is the odd cell out
        #                 scheiden ~ schiet ~ schieden   d ~ t ~ d
        #                 binden   ~ bant   ~ bunden     d ~ t ~ d
        #
        # So the two clauses below each carry a shape test:
        #
        # 1. A past sg ~ past pl difference is Verner only if the past singular
        #    still agrees with the present. This is the clause that recovers the
        #    Class IV and V Verner verbs, whose past ablaut was purely
        #    quantitative (a ~ â) and is invisible once vowel length is
        #    normalised away - and normalised away it must be, or the ReM/ReF
        #    transcription change at 1350 reads as a leveling event.
        #
        # 2. A present ~ past sg difference is Verner only if the past plural
        #    shares it, which puts the alternation in a medial position where
        #    devoicing cannot reach. This is what keeps snîden ~ sneit ~ sniten
        #    (d ~ t ~ t) and drops scheiden ~ schiet ~ schieden (d ~ t ~ d).
        #
        # The present ~ past pl pair needs no such test: both cells carry an
        # ending, so neither coda is word-final.
        #
        # Note what is deliberately absent: an "any GW anywhere and any ablaut
        # anywhere" clause. Nearly every strong verb has ablaut somewhere, so
        # that condition reduces to "any consonant difference at all" and hands
        # the treatment variable over to extraction noise.
        # Both shape tests turn on "the deciding cell does NOT differ", and
        # check_alternation returns False for a pair it cannot see. A missing
        # anchor therefore satisfies the test for want of data rather than on
        # the evidence, which is how scheiden d ~ t ~ ? passes as Verner
        # precisely because its past plural is absent. Each test asks for its
        # deciding cell separately.
        #
        # For the medial test the deciding cell is the past plural: it is what
        # says the alternation is not confined to word-final position.
        #
        # For the Verner test the deciding cell is the present, which is what
        # separates quëden t ~ t ~ d from scheiden d ~ t ~ d. That one is only
        # needed when devoicing could have produced the sg ~ pl contrast in the
        # first place. are_cons_equivalent already absorbs p ~ b and k ~ g, so
        # the dentals are the only voicing pair that still reaches here - kept
        # apart on purpose, because d ~ t is the Class I alternation. An s ~ r
        # or h ~ g contrast is not something Auslautverhärtung can make, so it
        # needs no present-tense witness: this is what lets wesen through on
        # was ~ wâren alone.
        pastpl_coda_known = _coda_known(row, "pastpl")
        pres_coda_known = _coda_known(row, "pres")
        sg_pl_is_dental = _last_chars(row, "pastsg", "pastpl") in DEVOICING_PAIRS

        gw_verner_sg_pl = (
            gw_sg_pl
            and not gw_pres_sg
            and (pres_coda_known or not sg_pl_is_dental)
        )
        gw_paradigmatic_pres_sg = gw_pres_sg and pastpl_coda_known and not gw_sg_pl

        is_bipartite = (
            (ab_pres_sg and gw_paradigmatic_pres_sg)
            or (ab_pres_pl and gw_pres_pl)
            or gw_verner_sg_pl
        )

        return pd.Series(
            [
                ab_pres_sg,
                gw_pres_sg,
                ab_pres_pl,
                gw_pres_pl,
                ab_sg_pl,
                gw_sg_pl,
                1 if is_bipartite else 0,
            ]
        )

    cols = [
        "diff_vowel_pres_pastsg",
        "diff_cons_pres_pastsg",
        "diff_vowel_pres_pastpl",
        "diff_cons_pres_pastpl",
        "diff_vowel_pastsg_pastpl",
        "diff_cons_pastsg_pastpl",
        "is_bipartite",
    ]

    anchors_pivoted[cols] = anchors_pivoted.apply(analyze_paradigm, axis=1)

    return anchors_pivoted


def step_3_establish_targets(
    df, nhg_file="data/lemmas/nhg_targets.csv", lemma_index=None
):
    """
    Identifies the 'Teleological Target' (End State) for both Present and Past.
    Returns a DataFrame unique by (lemma_id, variety).

    The target for a tense is the modal vowel and coda at the most recent ENHG
    date where that tense gives an extractable form. The search starts at the
    latest date and moves backwards only while the tense gives nothing.

    Note that the search does not pool dates. Each target comes from one date,
    which keeps the endpoint semantics of the target. A wider window would mix
    forms from before and after the leveling event. The mode could then land on
    the pre-leveling vowel, which inverts the direction of the comparison in
    step 4: the cell that carries the change becomes uninformative, and the
    cell that did not change gets coded as leveled.

    target_pres_date, target_past_date, target_pres_n and target_past_n record
    the date and the token count that each target rests on. A target from one
    token is weak evidence. These columns make that visible to the diagnostics.
    """
    print("\n--- Step 3: Determining Teleological Targets (ENHG) ---")

    # Modern German forms, where a curated one exists. These take priority over
    # anything the corpus can offer, because the corpus is too thin at the late
    # end to name an endpoint. The corpus rule stays as the fallback.
    if lemma_index is None:
        lemma_index = build_lemma_index(df)

    nhg = {}
    if os.path.exists(nhg_file):
        table = pd.read_csv(nhg_file, dtype=str).fillna("")
        # nhg_infinitive is an infinitive and nhg_preterite a 3rd singular
        # (barg, brach, fand), so they belong to different slots and the -n rule
        # has to see that.
        for _, entry in table.iterrows():
            parsed = {}
            for tense, column, slot in (
                ("pres", "nhg_infinitive", "Pres"),
                ("past", "nhg_preterite", "PastSg"),
            ):
                form = entry[column].strip()
                if not form:
                    continue
                vowel, coda = extract_root_structure(
                    df,
                    form,
                    corpus="ENHG",
                    lemma_id=entry["lemma_id"],
                    std_infl=slot,
                    lemma_index=lemma_index,
                )
                if not pd.isna(vowel):
                    parsed[tense] = (vowel, coda)
            if parsed:
                nhg[str(entry["lemma_id"])] = parsed
        print(f"Loaded modern forms for {len(nhg)} lemmas from {nhg_file}.")

    enhg_df = df[df["corpus"] == "ENHG"].copy()

    # Storage for the winning forms
    target_data = []

    def latest_mode(group, infl_values):
        """
        Walk the dates of one tense from latest to earliest. Return the vowel
        and coda modes from the first date that gives a mode, with that date
        and its token count.

        The vowel and the coda are resolved separately. Extraction can fail for
        one and not the other, and a shared date would discard the good one.
        """
        rows = group[group["std_infl"].isin(infl_values)]
        if rows.empty:
            return pd.NA, pd.NA, pd.NA, 0

        vowel, coda = pd.NA, pd.NA
        vowel_date, coda_date = pd.NA, pd.NA
        vowel_n, coda_n = 0, 0

        for date in sorted(rows["date"].dropna().unique(), reverse=True):
            dated = rows[rows["date"] == date]

            if pd.isna(vowel):
                values = dated["extracted_vowel"].dropna()
                mode = values.mode()
                if not mode.empty:
                    vowel, vowel_date, vowel_n = mode.iloc[0], date, len(values)

            if pd.isna(coda):
                values = dated["extracted_coda"].dropna()
                mode = values.mode()
                if not mode.empty:
                    coda, coda_date, coda_n = mode.iloc[0], date, len(values)

            if not pd.isna(vowel) and not pd.isna(coda):
                break

        # Report the date of the vowel target, or the coda target when the
        # vowel gives nothing. The vowel drives the majority of the coding.
        date = vowel_date if not pd.isna(vowel_date) else coda_date
        return vowel, coda, date, max(vowel_n, coda_n)

    # Group by Lemma and Variety to process each verb's history
    # We want ONE row per verb with: Target_Pres, Target_Past
    groups = enhg_df.groupby(["lemma_id", "variety"])

    for (lid, var), group in tqdm(groups, desc="Calculating Targets"):
        if group.empty:
            continue

        t_pres_v, t_pres_c, pres_date, pres_n = latest_mode(group, ["Pres"])
        t_past_v, t_past_c, past_date, past_n = latest_mode(
            group, ["PastSg", "PastPl"]
        )

        # A curated modern form replaces the corpus target for that tense.
        modern = nhg.get(str(lid), {})
        pres_source = past_source = "corpus"
        if "pres" in modern:
            t_pres_v, t_pres_c = modern["pres"]
            pres_source = "nhg"
        if "past" in modern:
            t_past_v, t_past_c = modern["past"]
            past_source = "nhg"

        target_data.append(
            {
                "lemma_id": lid,
                "variety": var,
                "target_vowel_pres": t_pres_v,
                "target_coda_pres": t_pres_c,
                "target_vowel_past": t_past_v,
                "target_coda_past": t_past_c,
                "target_pres_date": pres_date,
                "target_pres_n": pres_n,
                "target_past_date": past_date,
                "target_past_n": past_n,
                "target_pres_source": pres_source,
                "target_past_source": past_source,
            }
        )

    # The loop above walks ENHG groups, so a lemma that ReF never attests never
    # reaches it - and the curated modern form, which is applied inside the
    # loop, never reaches that lemma either. That gate belongs on the corpus
    # fallback alone: the endpoint of kiesen is kor whether or not ReF happens
    # to write the verb down. Verbs whose modern reflex the curator could not
    # find are unaffected, because they have no curated form to carry.
    curated_only = []
    if nhg:
        covered = {(str(entry["lemma_id"]), entry["variety"]) for entry in target_data}
        seen = df[["lemma_id", "variety"]].dropna().drop_duplicates()
        for _, place in seen.iterrows():
            lid, var = place["lemma_id"], place["variety"]
            if (str(lid), var) in covered:
                continue
            modern = nhg.get(str(lid))
            if not modern:
                continue
            curated_only.append(
                {
                    "lemma_id": lid,
                    "variety": var,
                    "target_vowel_pres": modern.get("pres", (pd.NA, pd.NA))[0],
                    "target_coda_pres": modern.get("pres", (pd.NA, pd.NA))[1],
                    "target_vowel_past": modern.get("past", (pd.NA, pd.NA))[0],
                    "target_coda_past": modern.get("past", (pd.NA, pd.NA))[1],
                    # No ENHG attestation stands behind these, so there is no
                    # corpus date or token count to report.
                    "target_pres_date": pd.NA,
                    "target_pres_n": 0,
                    "target_past_date": pd.NA,
                    "target_past_n": 0,
                    "target_pres_source": "nhg" if "pres" in modern else "none",
                    "target_past_source": "nhg" if "past" in modern else "none",
                }
            )
    if curated_only:
        print(
            f"Carried a curated modern target to {len(curated_only)} lemma-variety "
            f"groups that ReF never attests."
        )
    target_data.extend(curated_only)

    result = pd.DataFrame(target_data)

    resolved_past = result["target_vowel_past"].notna().sum()
    resolved_pres = result["target_vowel_pres"].notna().sum()
    print(
        f"Resolved a past vowel target for {resolved_past} of {len(result)} "
        f"lemma-variety groups, and a present vowel target for {resolved_pres}."
    )
    for tense in ("pres", "past"):
        counts = result[f"target_{tense}_source"].value_counts().to_dict()
        print(f"  {tense} target source: {counts}")
    return result


def step_4_coding_outcome(df, baseline_df, target_df, sc_file="data/vowel_changes.csv"):
    print(
        "\n--- Step 4: Coding Leveling Events (With Historical Alternation Filters) ---"
    )

    sc_dict = load_sound_changes(sc_file)

    # Merge Data
    main = df.merge(baseline_df, on=["lemma_id", "variety"], how="left")
    main = main.merge(target_df, on=["lemma_id", "variety"], how="left")

    analyzable = main[
        (main["std_infl"].isin(["PastSg", "PastPl"]))
        & (main["extracted_vowel"].notna())
    ].copy()

    def code_row(row):
        variety = row["variety"]
        infl = row["std_infl"]  # Current observation slot (PastSg or PastPl)

        # Anchors
        anchor_pres_v = row.get("anchor_vowel_pres")
        anchor_pres_c = row.get("anchor_coda_pres")

        if infl == "PastSg":
            anchor_self_v = row.get("anchor_vowel_pastsg")
            anchor_self_c = row.get("anchor_coda_pastsg")
            anchor_other_v = row.get("anchor_vowel_pastpl")
            anchor_other_c = row.get("anchor_coda_pastpl")

            # --- FLAGGING FLAGS (For PastSg) ---
            # Did I differ from Present?
            hist_diff_v_pres = row.get("diff_vowel_pres_pastsg")
            hist_diff_c_pres = row.get("diff_cons_pres_pastsg")
            # Did I differ from PastPl?
            hist_diff_v_other = row.get("diff_vowel_pastsg_pastpl")
            hist_diff_c_other = row.get("diff_cons_pastsg_pastpl")

        else:  # PastPl
            anchor_self_v = row.get("anchor_vowel_pastpl")
            anchor_self_c = row.get("anchor_coda_pastpl")
            anchor_other_v = row.get("anchor_vowel_pastsg")
            anchor_other_c = row.get("anchor_coda_pastsg")

            # --- FLAGGING FLAGS (For PastPl) ---
            # Did I differ from Present?
            hist_diff_v_pres = row.get("diff_vowel_pres_pastpl")
            hist_diff_c_pres = row.get("diff_cons_pres_pastpl")
            # Did I differ from PastSg?
            hist_diff_v_other = row.get("diff_vowel_pastsg_pastpl")
            hist_diff_c_other = row.get("diff_cons_pastsg_pastpl")

        # Contrasts the pre-1200 baseline established for this paradigm. Each
        # one is protected from the sound-change filter, in both directions.
        def build_protected(pairs):
            out = set()
            for first, second, differs in pairs:
                if differs is not True:
                    continue
                if pd.isna(first) or pd.isna(second):
                    continue
                first, second = str(first), str(second)
                if first != second:
                    out.add(frozenset((first, second)))
            return out

        protected_v = build_protected(
            [
                (anchor_pres_v, anchor_self_v, hist_diff_v_pres),
                (anchor_self_v, anchor_other_v, hist_diff_v_other),
            ]
        )
        # Codas are compared on their last character, so protect that form too.
        def last_char(value):
            if pd.isna(value):
                return value
            text = str(value).lower()
            return text[-1] if len(text) > 1 else text

        protected_c = build_protected(
            [
                (last_char(anchor_pres_c), last_char(anchor_self_c), hist_diff_c_pres),
                (last_char(anchor_self_c), last_char(anchor_other_c), hist_diff_c_other),
            ]
        )

        obs_v = row["extracted_vowel"]
        obs_c = row["extracted_coda"]

        if pd.isna(anchor_self_v):
            return pd.Series([pd.NA] * 8)

        # --- Helper for Comparison ---
        def get_status(
            observed,
            anchor_self,
            anchor_compare,
            target,
            is_cons,
            historical_diff_exists,
        ):
            """
            historical_diff_exists: Boolean from Step 2.
            If False, we return NA because there is no alternation to level.
            """
            # 1. GATEKEEPER CHECK
            if historical_diff_exists is False:  # Explicit False check (not NA)
                return pd.NA

            if pd.isna(target) or pd.isna(anchor_compare):
                return pd.NA

            def is_equiv(a, b):
                if is_cons:
                    return are_cons_equivalent(a, b, protected=protected_c)
                else:
                    return are_vowels_equivalent(
                        a, b, variety, sc_dict, protected=protected_v
                    )

            # 2. Target Validity Check
            # If the Target == Anchor (via sound change), we can't detect leveling.
            if is_equiv(target, anchor_self):
                return pd.NA

            # 3. Outcome Check
            is_match_target = is_equiv(observed, target)
            is_stay_anchor = is_equiv(observed, anchor_self)

            if is_match_target:
                return 1  # Leveled
            elif is_stay_anchor:
                return 0  # Resisted

            return pd.NA

        def get_alternation_pair(anchor_self, anchor_compare, historical_diff_exists):
            if historical_diff_exists is False:
                return pd.NA
            if pd.isna(anchor_self) or pd.isna(anchor_compare):
                return pd.NA
            return f"{anchor_self} ~ {anchor_compare}"

        # --- Scenario 1: Leveling to Present (Weakening/Analogical leveling) ---
        # We pass 'hist_diff_v_pres' to ensure we only count leveling if Pres and Past differed historically
        lv_pres = get_status(
            obs_v,
            anchor_self_v,
            anchor_pres_v,
            row.get("target_vowel_pres"),
            False,
            hist_diff_v_pres,
        )
        lc_pres = get_status(
            obs_c,
            anchor_self_c,
            anchor_pres_c,
            row.get("target_coda_pres"),
            True,
            hist_diff_c_pres,
        )
        alt_v_pres = get_alternation_pair(anchor_self_v, anchor_pres_v, hist_diff_v_pres)
        alt_c_pres = get_alternation_pair(anchor_self_c, anchor_pres_c, hist_diff_c_pres)

        # --- Scenario 2: Leveling to Past (Internal Simplification) ---
        # We pass 'hist_diff_v_other' to ensure we only count leveling if Sg and Pl differed historically
        lv_past = get_status(
            obs_v,
            anchor_self_v,
            anchor_other_v,
            row.get("target_vowel_past"),
            False,
            hist_diff_v_other,
        )
        lc_past = get_status(
            obs_c,
            anchor_self_c,
            anchor_other_c,
            row.get("target_coda_past"),
            True,
            hist_diff_c_other,
        )
        alt_v_past = get_alternation_pair(anchor_self_v, anchor_other_v, hist_diff_v_other)
        alt_c_past = get_alternation_pair(anchor_self_c, anchor_other_c, hist_diff_c_other)

        return pd.Series([lv_pres, lv_past, lc_pres, lc_past, alt_v_pres, alt_v_past, alt_c_pres, alt_c_past])

    # Apply
    cols = [
        "is_leveled_vowel_pres",
        "is_leveled_vowel_past",
        "is_leveled_cons_pres",
        "is_leveled_cons_past",
        "vowel_alternation_pres",
        "vowel_alternation_past",
        "cons_alternation_pres",
        "cons_alternation_past",
    ]

    tqdm.pandas(desc="Coding Variables")
    analyzable[cols] = analyzable.progress_apply(code_row, axis=1)

    return analyzable


# ------------------------------------------------------------------------------
# 4. EXECUTION
# ------------------------------------------------------------------------------


def run_pipeline(
    input_file="data/combined_normalized_corpus.csv",
    output_file="data/coded_output.csv",
    nhg_file="data/lemmas/nhg_targets.csv",
):
    """
    nhg_file is a parameter so that a sensitivity run can point the modern
    targets at an alternative table without touching the one on disk. See
    analysis/marking_type_summary.py --sensitivity.
    """
    df = pd.read_csv(input_file, dtype=str)

    # The lemma strings carry the prefix analysis. Read it once and hand the
    # same index to every stage that extracts a root, so the corpus forms and
    # the modern target forms are segmented by the same rule.
    lemma_index = build_lemma_index(df)

    # 1. Preprocess & Extract Root Parts
    df_processed = step_1_preprocessing(df, lemma_index=lemma_index)

    # 2. Get Anchors (Start State)
    baseline_df = step_2_establish_baseline(df_processed)

    # 3. Get Targets (End State)
    target_df = step_3_establish_targets(
        df_processed, nhg_file=nhg_file, lemma_index=lemma_index
    )

    # 4. Code Variables
    final_df = step_4_coding_outcome(df_processed, baseline_df, target_df)

    print(f"Saving {len(final_df)} coded rows to {output_file}...")
    final_df.to_csv(output_file, index=False)
    return final_df


if __name__ == "__main__":
    df = run_pipeline()
