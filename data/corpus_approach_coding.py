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
    """Normalizes inflection labels to the three critical slots."""
    if pd.isna(val):
        return "Pres"
    val = str(val).lower()
    if "participle" in val or "ppl" in val:
        return "Ppl"
    if "past" in val or "prät" in val or "pret" in val:
        if "pl" in val or "2" in val:  # 2nd person often patterns with Plural in MHG
            return "PastPl"
        else:
            return "PastSg"
    return "Pres"


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

    # 4. Final filter for valid alphabet (Optional, but safe)
    # Keep standard German alphabet + common IPA chars you use
    # Note: \u00DF is ß
    f = re.sub(r"[^a-zäöüßſ\u00E6\u0153\u0283\u02A6\u03C7]", "", f)

    return f


def extract_root_structure(df, form, corpus="MHG", lemma_id=None):
    """
    Robust extraction with Phonotactic Guards to prevent root-eating.
    """
    f = clean_form(form)
    if not f:
        return pd.NA, pd.NA

    # --- 1. SMART PREFIX STRIPPING (PHONOTACTIC AWARE) ---
    prefix_list = list(PREFIXES.get(corpus, PREFIXES["MHG"]))

    # We sort prefixes by length so we try longest matches first
    # But we iterate repeatedly to handle recursive prefixes (e.g. vor-ge-schrieben)
    clean_stem = f

    # Safety loop limit to prevent infinite loops (though unlikely)
    for _ in range(3):
        match_found = False
        for p in prefix_list:
            if clean_stem.startswith(p):
                candidate = clean_stem[len(p) :]

                # --- PHONOTACTIC GUARD ---
                # Check 1: Vowel Existence
                # If remainder has no vowels, we definitely stripped the nucleus. Stop.
                if not any(v in candidate for v in VOWELS):
                    continue

                # Check 2: Onset Validity
                # Extract the onset (consonants before the first vowel)
                # We use the normalized vowel set for regex matching
                v_pattern = f"[{''.join(VOWELS)}]"
                onset_match = re.search(f"^([^{''.join(VOWELS)}]+)", candidate)

                if onset_match:
                    onset = onset_match.group(1)

                    # Guard A: Geminates (e.g., 'zz', 'ss', 'mm')
                    # Roots never start with geminates. If we see one, we stripped 'ge' from 'gessen'.
                    if len(onset) > 1 and onset[0] == onset[1]:
                        continue  # Invalid strip

                    # Guard B: Illegal Clusters (e.g., 'lt', 'rg', 'nc')
                    # If onset is a cluster (>1 char) and NOT in our whitelist, it's invalid.
                    # Note: 'sch' is 3 chars, but in VALID_ONSETS. 'lt' is not.
                    if len(onset) > 1 and onset not in VALID_ONSETS:
                        # Edge case: 'sch' might be parsed as 's', 'c', 'h' if not careful
                        # But our onset is raw string.
                        # We must check if the *start* of the onset is a valid digraph/trigraph
                        # or if the whole onset is allowed.

                        # Simple check: Is the specific cluster in the allowed list?
                        # Or does it start with a valid 3-char (sch) or 2-char onset?
                        is_valid_cluster = False
                        if onset in VALID_ONSETS:
                            is_valid_cluster = True
                        else:
                            # Check prefixes of the onset (e.g. 'sch' in 'schri...')
                            # Actually, regex `^[^vowels]+` grabs all initial consonants.
                            # 'schreiben' -> onset 'schr'. 'schr' not in list.
                            # But 'schr' starts with 'sch' (valid) + 'r'.
                            # German allows complex onsets like 'schr', 'spr', 'str'.
                            # Let's simplify:
                            # If it starts with an illegal PAIR, reject.
                            # Illegal pairs: 'lt', 'rg', 'nt', 'mp'.
                            # Legal starts: valid singletons or VALID_ONSETS.
                            pass  # Too complex for simple logic?

                        # REVISED GUARD B:
                        # Just block specific known "Coda" clusters that appear after 'ge-'/'be-' stripping
                        # Common culprits: lt (gelten), rg (bergen), rc (borc), rb (sterben)
                        if onset.startswith(
                            (
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
                        ):
                            continue

                # If we passed guards, accept the strip
                clean_stem = candidate
                match_found = True
                break  # Restart loop to check for next prefix (e.g. vor-ge-)

        if not match_found:
            break

    # --- 2. EXTRACT NUCLEUS ---
    v_pattern = f"[{''.join(VOWELS)}]+"
    match = re.search(f"({v_pattern})", clean_stem)

    if not match:
        return pd.NA, pd.NA

    nucleus = match.group(1)
    post_nucleus = clean_stem[match.end() :]

    # --- 3. EXTRACT RAW CODA ---
    coda_match = re.search(f"^([^{''.join(VOWELS)}]+)", post_nucleus)
    final_coda = coda_match.group(1) if coda_match else ""

    # --- 4. PROTECTED SUFFIX STRIPPING ---
    sorted_suffixes = sorted(SUFFIXES, key=len, reverse=True)

    for s in sorted_suffixes:
        if final_coda.endswith(s):
            # Constraint: Don't strip if it leaves the coda empty
            # UNLESS the remaining stem implies a vowel-final root (like 'schrien').
            # Heuristic: If stripping 'n' leaves empty, allow it ONLY if nucleus is a diphthong/long vowel?
            # Safer: Just STRICTLY enforce "Root must have coda" for strong verbs?
            # Most strong verbs end in C. Exceptions: gan, stan, tuon, schrien, spien.
            # If we enforce len > 0, we break 'schrien'.
            # If we allow len == 0, we might break 'geschehen' (h -> empty).

            # SPECIFIC FIX FOR 'GESCHEHEN' (Coda 'h') vs Suffix 'en'
            # 'h' does not end with 'en'. So 'h' is safe.
            # The previous issue was likely 'schehen' -> 'h' -> interpreted as empty?

            if len(final_coda) > len(s):
                final_coda = final_coda[: -len(s)]
                break
            elif len(final_coda) == len(s):
                # Only strip entire coda if it is exactly 'n' or 'en' (common infinitive markers on vowel roots)
                if s in ["n", "en"]:
                    final_coda = ""
                    break
                # Otherwise protect it (e.g. 't' in 'gilt')
                pass

    # --- 5. CLEANUP ---
    for k, v in DIGRAPH_IPA.items():
        final_coda = final_coda.replace(k, v)
        nucleus = nucleus.replace(k, v)

    final_coda = re.sub(r"(.)\1+", r"\1", final_coda)  # Degeminate

    return nucleus, final_coda


# ------------------------------------------------------------------------------
# 3. ANALYSIS PIPELINE
# ------------------------------------------------------------------------------


def step_1_preprocessing(df):
    print("--- Step 1: Preprocessing & Root Extraction (Lemma-Aware) ---")
    df = df.copy()

    df = df[~df["inflClass"].isin(["St|Sw|Unr", "Unr", "irr", "St|Unr", "prpr"])]

    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df["std_infl"] = df["infl"].apply(standardize_infl)

    # We can no longer just map unique forms blindly, because 'geben' (Inf) and
    # 'geben' (some other context) might differ if we use lemma info.
    # However, for speed, we can group by (Form, Corpus, Lemma).

    unique_contexts = df[["norm", "corpus", "lemma_id"]].drop_duplicates()

    print(f"   Extracting roots for {len(unique_contexts)} unique form contexts...")

    results = []
    for _, row in tqdm(unique_contexts.iterrows(), total=len(unique_contexts)):
        # PASS THE LEMMA HERE
        v, c = extract_root_structure(df, row["norm"], row["corpus"], row["lemma_id"])
        results.append(
            {
                "norm": row["norm"],
                "corpus": row["corpus"],
                "lemma_id": row["lemma_id"],
                "extracted_vowel": v,
                "extracted_coda": c,
            }
        )

    # Map back
    res_df = pd.DataFrame(results)
    df = df.merge(res_df, on=["norm", "corpus", "lemma_id"], how="left")

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

        # Global Bipartite Definition:
        # If ANY of the three pairs shows BOTH Ablaut AND GW.
        is_bipartite = (
            (ab_pres_sg and gw_pres_sg)
            or (ab_pres_pl and gw_pres_pl)
            or (ab_sg_pl and gw_sg_pl)
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


def step_3_establish_targets(df, nhg_file="data/lemmas/nhg_targets.csv"):
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
    nhg = {}
    if os.path.exists(nhg_file):
        table = pd.read_csv(nhg_file, dtype=str).fillna("")
        for _, entry in table.iterrows():
            parsed = {}
            for tense, column in (("pres", "nhg_infinitive"), ("past", "nhg_preterite")):
                form = entry[column].strip()
                if not form:
                    continue
                vowel, coda = extract_root_structure(
                    df, form, corpus="ENHG", lemma_id=entry["lemma_id"]
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

    # 1. Preprocess & Extract Root Parts
    df_processed = step_1_preprocessing(df)

    # 2. Get Anchors (Start State)
    baseline_df = step_2_establish_baseline(df_processed)

    # 3. Get Targets (End State)
    target_df = step_3_establish_targets(df_processed, nhg_file=nhg_file)

    # 4. Code Variables
    final_df = step_4_coding_outcome(df_processed, baseline_df, target_df)

    print(f"Saving {len(final_df)} coded rows to {output_file}...")
    final_df.to_csv(output_file, index=False)
    return final_df


if __name__ == "__main__":
    df = run_pipeline()
