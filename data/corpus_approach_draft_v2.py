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
    {"t", "d"},  # Devoicing
    {"k", "g", "c", "q"},  # Devoicing / Spelling
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


def are_cons_equivalent(c1, c2):
    """
    Returns True if c1 and c2 are phonologically or orthographically equivalent
    (i.e., NOT Grammatischer Wechsel).
    """
    if pd.isna(c1) or pd.isna(c2):
        return False

    c1, c2 = str(c1).lower(), str(c2).lower()
    c1 = c1[-1] if len(c1) > 1 else c1
    c2 = c2[-1] if len(c2) > 1 else c2
    if c1 == c2:
        return True
    for s in EQUIV_SETS:
        if c1 in s and c2 in s:
            return True
    return False


def are_vowels_equivalent(v1, v2, variety, sc_dict):
    """
    Returns True if v1 == v2 OR if the transition v1->v2 (or v2->v1)
    is a regular sound change in the given variety.
    """
    if pd.isna(v1) or pd.isna(v2):
        return False

    v1, v2 = str(v1), str(v2)

    # 1. Strict Identity
    if v1 == v2:
        return True

    # 2. Sound Change Lookup
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


def extract_root_structure(form, corpus="MHG", lemma=None):
    """
    Robust extraction that protects root consonants and handles 'ge-' lemmas.
    """
    f = clean_form(form)
    if not f:
        return pd.NA, pd.NA

    # --- 1. SMART PREFIX STRIPPING ---
    prefix_list = list(PREFIXES.get(corpus, PREFIXES["MHG"]))  # Make a copy

    # Safety Check: If the lemma ITSELF starts with 'ge' (e.g., geben, gewinnen),
    # we should NOT strip 'ge' from the form unless it appears twice (ge-geben).
    lemma_starts_with_ge = False
    if lemma and str(lemma).lower().startswith("ge"):
        lemma_starts_with_ge = True
        if "ge" in prefix_list:
            prefix_list.remove("ge")

    # Special handling for Participles of ge-roots (e.g., 'gegeben')
    # If we removed 'ge' from the list, 'gegeben' won't strip. We must manually handle the first 'ge'.
    if lemma_starts_with_ge and f.startswith("gege"):
        f = f[2:]  # Strip the inflectional 'ge-', leave the radical 'ge-'

    clean_stem = f
    match_found = True
    while match_found:
        match_found = False
        for p in prefix_list:
            if clean_stem.startswith(p):
                # LOOKAHEAD SAFETY:
                # Only strip if the remainder still looks like a valid root (has a vowel).
                remainder = clean_stem[len(p) :]

                # Check for vowels in remainder
                if any(v in remainder for v in VOWELS):
                    clean_stem = remainder
                    match_found = True
                    break

    # --- 2. EXTRACT NUCLEUS & RAW CODA ---
    v_pattern = f"[{''.join(VOWELS)}]+"
    match = re.search(f"({v_pattern})", clean_stem)

    if not match:
        return pd.NA, pd.NA

    nucleus = match.group(1)
    post_nucleus = clean_stem[match.end() :]

    # Capture immediate consonant cluster
    coda_match = re.search(f"^([^{''.join(VOWELS)}]+)", post_nucleus)
    final_coda = coda_match.group(1) if coda_match else ""

    # --- 3. PROTECTED SUFFIX STRIPPING ---
    # We iterate through suffixes, but we apply the "Non-Empty" Constraint
    sorted_suffixes = sorted(SUFFIXES, key=len, reverse=True)

    for s in sorted_suffixes:
        if final_coda.endswith(s):
            # CONSTRAINT: Only strip if we are left with at least 1 consonant.
            # Exception: If the suffix is distinct from the root (hard to know without dict),
            # strictly speaking, for Strong Verbs, the root usually ends in a consonant.
            # We assume if stripping 's' leaves 0 chars, 's' was likely the root consonant (e.g. 'lei-d-en' -> 'd').

            if len(final_coda) > len(s):
                final_coda = final_coda[: -len(s)]
                break  # We usually only strip the longest matching tail (e.g. 'est')
            else:
                # The suffix covers the entire Coda (e.g. 'gap' -> coda 'p', suffix 'p' not in list usually,
                # but 'leiden' -> coda 'd', suffix 'd' IS in list).
                # We do NOT strip. We assume this is the root.
                pass

    # --- 4. FINAL CLEANUP ---
    # Digraph normalization
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

    unique_contexts = df[["norm", "corpus", "lemma"]].drop_duplicates()

    print(f"   Extracting roots for {len(unique_contexts)} unique form contexts...")

    results = []
    for _, row in tqdm(unique_contexts.iterrows(), total=len(unique_contexts)):
        # PASS THE LEMMA HERE
        v, c = extract_root_structure(row["norm"], row["corpus"], row["lemma"])
        results.append(
            {
                "norm": row["norm"],
                "corpus": row["corpus"],
                "lemma": row["lemma"],
                "extracted_vowel": v,
                "extracted_coda": c,
            }
        )

    # Map back
    res_df = pd.DataFrame(results)
    df = df.merge(res_df, on=["norm", "corpus", "lemma"], how="left")

    return df


def step_2_establish_baseline(df):
    """
    Establishes the 'Start State' (MHG Pre-1150).
    Now captures Present, PastSg, and PastPl anchors to check for pre-existing alternations.
    """
    print("\n--- Step 2: Establishing Diachronic Baseline (Pre-1150) ---")

    # Filter for Baseline Candidates
    baseline_df = df[
        (df["date"] <= 1150) & (df["corpus"] == "MHG") & (df["extracted_vowel"].notna())
    ].copy()

    # We need anchors for ALL three slots to check alternations
    # Group by Lemma+Variety+Infl to get the mode
    anchors = (
        baseline_df.groupby(["lemma_id", "variety", "std_infl"])[
            ["extracted_vowel", "extracted_coda"]
        ]
        .agg(lambda x: pd.Series.mode(x)[0] if not x.mode().empty else pd.NA)
        .reset_index()
    )

    # Pivot to get one row per Lemma+Variety with columns for each slot
    # pivot_table might be messy with strings, so we do a manual unstack
    anchors_pivoted = anchors.pivot_table(
        index=["lemma_id", "variety"],
        columns="std_infl",
        values=["extracted_vowel", "extracted_coda"],
        aggfunc="first",  # Should be unique per group already
    )

    # Flatten MultiIndex columns (e.g., ('extracted_vowel', 'Pres') -> 'anchor_vowel_pres')
    anchors_pivoted.columns = [
        f"anchor_{x.split('_')[1]}_{y.lower()}" for x, y in anchors_pivoted.columns
    ]

    anchors_pivoted = anchors_pivoted.reset_index()

    # Calculate Complexity (Bipartite) for the Past Tense
    # We do this here so it's ready for the regression model later
    def calc_bipartite(row):
        # We need Sg and Pl to judge Bipartite
        sg_v, sg_c = row.get("anchor_vowel_pastsg"), row.get("anchor_coda_pastsg")
        pl_v, pl_c = row.get("anchor_vowel_pastpl"), row.get("anchor_coda_pastpl")

        if pd.isna(sg_v) or pd.isna(pl_v):
            return pd.NA

        has_ablaut = sg_v != pl_v
        # GW exists if consonants differ AND are not just spelling variants
        has_gw = (sg_c != pl_c) and (not are_cons_equivalent(sg_c, pl_c))

        return 1 if (has_ablaut and has_gw) else 0

    anchors_pivoted["is_bipartite"] = anchors_pivoted.apply(calc_bipartite, axis=1)

    return anchors_pivoted


def step_3_establish_targets(df):
    """
    Identifies the 'Teleological Target' (End State) for both Present and Past.
    Returns a DataFrame unique by (lemma_id, variety).
    """
    print("\n--- Step 3: Determining Teleological Targets (ENHG) ---")

    enhg_df = df[df["corpus"] == "ENHG"].copy()

    # Storage for the winning forms
    target_data = []

    # Group by Lemma and Variety to process each verb's history
    # We want ONE row per verb with: Target_Pres, Target_Past
    groups = enhg_df.groupby(["lemma_id", "variety"])

    for (lid, var), group in tqdm(groups, desc="Calculating Targets"):
        if group.empty:
            continue

        # 1. Establish the Target Date (Latest available date for this verb)
        max_date = group["date"].max()
        latest_rows = group[group["date"] == max_date]

        # 2. Identify the Present Tense Target
        # The most frequent form labeled 'Pres' at the latest date
        pres_rows = latest_rows[latest_rows["std_infl"] == "Pres"]
        if not pres_rows.empty:
            pres_v_mode = pres_rows["extracted_vowel"].dropna().mode()
            pres_c_mode = pres_rows["extracted_coda"].dropna().mode()
            t_pres_v = pres_v_mode.iloc[0] if not pres_v_mode.empty else pd.NA
            t_pres_c = pres_c_mode.iloc[0] if not pres_c_mode.empty else pd.NA
        else:
            t_pres_v, t_pres_c = pd.NA, pd.NA

        # 3. Identify the Past Tense Target
        past_rows = latest_rows[latest_rows["std_infl"].isin(["PastSg", "PastPl"])]
        if not past_rows.empty:
            past_v_mode = past_rows["extracted_vowel"].dropna().mode()
            past_c_mode = past_rows["extracted_coda"].dropna().mode()
            t_past_v = past_v_mode.iloc[0] if not past_v_mode.empty else pd.NA
            t_past_c = past_c_mode.iloc[0] if not past_c_mode.empty else pd.NA
        else:
            t_past_v, t_past_c = pd.NA, pd.NA

        target_data.append(
            {
                "lemma_id": lid,
                "variety": var,
                "target_vowel_pres": t_pres_v,
                "target_coda_pres": t_pres_c,
                "target_vowel_past": t_past_v,
                "target_coda_past": t_past_c,
            }
        )

    return pd.DataFrame(target_data)


def step_4_coding_outcome(df, baseline_df, target_df, sc_file="data/vowel_changes.csv"):
    print("\n--- Step 4: Coding Leveling Events (With Sound Change Filter) ---")

    # 1. Load Sound Changes
    sc_dict = load_sound_changes(sc_file)

    # 2. Merge Data
    main = df.merge(baseline_df, on=["lemma_id", "variety"], how="left")
    main = main.merge(target_df, on=["lemma_id", "variety"], how="left")

    analyzable = main[
        (main["std_infl"].isin(["PastSg", "PastPl"]))
        & (main["extracted_vowel"].notna())
    ].copy()

    def code_row(row):
        variety = row["variety"]
        infl = row["std_infl"]

        # --- A. Get Anchors (Start State) ---
        # Present Anchor (for "Did it become Weak?" check)
        anchor_pres_v = row.get("anchor_vowel_pres")
        anchor_pres_c = row.get("anchor_coda_pres")

        # Past Anchors (for "Did it simplify?" check)
        if infl == "PastSg":
            anchor_self_v = row.get("anchor_vowel_pastsg")
            anchor_self_c = row.get("anchor_coda_pastsg")
            anchor_other_v = row.get("anchor_vowel_pastpl")  # The paradigm mate
            anchor_other_c = row.get("anchor_coda_pastpl")
        else:  # PastPl
            anchor_self_v = row.get("anchor_vowel_pastpl")
            anchor_self_c = row.get("anchor_coda_pastpl")
            anchor_other_v = row.get("anchor_vowel_pastsg")
            anchor_other_c = row.get("anchor_coda_pastsg")

        obs_v = row["extracted_vowel"]
        obs_c = row["extracted_coda"]

        if pd.isna(anchor_self_v):
            return pd.Series([pd.NA] * 4)

        # --- B. The Decision Logic ---
        def get_status(observed, anchor_self, anchor_compare, target, is_cons):
            """
            observed:       The form in text (ENHG)
            anchor_self:    The MHG origin form
            anchor_compare: The MHG form we are checking against (Pre-existing alternation check)
            target:         The ENHG attractor
            """
            if pd.isna(target) or pd.isna(anchor_compare):
                return pd.NA

            # Comparison Helper
            def is_equiv(a, b):
                if is_cons:
                    return are_cons_equivalent(a, b)
                else:
                    return are_vowels_equivalent(a, b, variety, sc_dict)

            # 1. Existence Check: Did an alternation exist in the first place?
            # If Self and Compare are functionally equivalent (via sound change or spelling),
            # there was no conflict to resolve.
            if is_equiv(anchor_self, anchor_compare):
                return pd.NA

            # 2. Target Check: Did the Target actually change from the Anchor?
            # If Target == Anchor (via sound change), then 'leveling' to target is just 'staying same'.
            # We treat this as valid data, but we must ensure we don't misclassify it.
            # If target is equiv to anchor, then `is_match_target` and `is_stay_anchor` will BOTH be True.
            # In that case, we should return NA (Stability), because we can't prove leveling happened.
            if is_equiv(target, anchor_self):
                return pd.NA

            # 3. Outcome Check
            is_match_target = is_equiv(observed, target)
            is_stay_anchor = is_equiv(observed, anchor_self)

            if is_match_target:
                return 1  # Leveled
            elif is_stay_anchor:
                return 0  # Resisted

            return pd.NA  # Ambiguous

        # --- C. Scenario 1: Leveling to Present (Weakening) ---
        lv_pres = get_status(
            obs_v, anchor_self_v, anchor_pres_v, row.get("target_vowel_pres"), False
        )
        lc_pres = get_status(
            obs_c, anchor_self_c, anchor_pres_c, row.get("target_coda_pres"), True
        )

        # --- D. Scenario 2: Leveling to Past (Internal Simplification) ---
        lv_past = get_status(
            obs_v, anchor_self_v, anchor_other_v, row.get("target_vowel_past"), False
        )
        lc_past = get_status(
            obs_c, anchor_self_c, anchor_other_c, row.get("target_coda_past"), True
        )

        return pd.Series([lv_pres, lv_past, lc_pres, lc_past])

    # Apply
    cols = [
        "is_leveled_vowel_pres",
        "is_leveled_vowel_past",
        "is_leveled_cons_pres",
        "is_leveled_cons_past",
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
):
    df = pd.read_csv(input_file, dtype=str)

    # 1. Preprocess & Extract Root Parts
    df_processed = step_1_preprocessing(df)

    # 2. Get Anchors (Start State)
    baseline_df = step_2_establish_baseline(df_processed)

    # 3. Get Targets (End State)
    target_df = step_3_establish_targets(df_processed)

    # 4. Code Variables
    final_df = step_4_coding_outcome(df_processed, baseline_df, target_df)

    print(f"Saving {len(final_df)} coded rows to {output_file}...")
    final_df.to_csv(output_file, index=False)

    return final_df


df = run_pipeline()

df_notna = df[
    ((~df["is_leveled_vowel_pres"].isna()) | (~df["is_leveled_cons_pres"].isna()))
    & (~df["is_bipartite"].isna())
]
