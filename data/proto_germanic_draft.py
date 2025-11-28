import json
import requests
import time
from tqdm import tqdm
import urllib.parse
import lxml.html
import re
import pandas as pd

# Taken from previous study, has all PG strong verbs listed
with open("data/cognates_germanic.json", encoding="UTF-8") as f:
    cognates = json.load(f)

# Needed to extract the list of verb lemmas we have
with open("data/lemmas/enhg_mapping.json", encoding="UTF-8") as f:
    enhg_lemmas = json.load(f)

# Current corpus data
corpus_df = pd.read_csv("data/combined_normalized_corpus.csv")

high_german_dict = dict()

for key in cognates.keys():
    for cognate_lst in cognates[key]:
        lang, form, _ = cognate_lst
        if lang == "de":
            high_german_dict[form] = key


# keep results: high german lemma -> (proto-germanic-lemma, 3pp-past-form-or-None)
hg_to_proto_with_conj = {}

session = requests.Session()
session.headers.update({"User-Agent": "Chrome/5.0 (compatible; script/1.0)"})

for hg_lemma, proto_lemma in tqdm(high_german_dict.items(), desc="Extracting PG forms"):
    url_fragment = urllib.parse.quote(proto_lemma, safe="")
    url = f"https://en.wiktionary.org/wiki/Reconstruction:Proto-Germanic/{url_fragment}"
    conj_form = None
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        doc = lxml.html.fromstring(resp.content)
        # primary XPath provided; try with and without tbody
        ### XPath for 3rd Person Plural Past
        nodes = doc.xpath(
            '//*[@id="mw-content-text"]/div[1]/div[6]/table/tbody/tr[20]/td[1]/span/a'
        )
        if not nodes:
            nodes = doc.xpath(
                '//*[@id="mw-content-text"]/div[1]/div[6]/table//tr[20]/td[1]/span/a'
            )
        if nodes:
            conj_form = nodes[0].text_content().strip()
    except Exception as e:
        continue

    if not conj_form:
        continue

    hg_to_proto_with_conj[hg_lemma] = (proto_lemma, conj_form)
    time.sleep(0.5)

mappable_lemmas = []
for lemma in set(enhg_lemmas.values()):
    if lemma in hg_to_proto_with_conj.keys():
        mappable_lemmas.append(lemma)


def analyze_pgmc_pair(inf, past_pl):
    """
    Compares PGmc Infinitive vs Past Plural to detect Verner and Ablaut.
    Modifications: Ignores 'w' and compares only the final single consonant.
    """
    # 1. Clean strings
    inf = inf.replace("*", "").lower().strip()
    past = past_pl.replace("*", "").lower().strip()

    # 2. Strip Suffixes to get Stem
    # Remove infinitive ending (-aną, -janą, -naną)
    root_inf = re.sub(r"(j|n)?aną$", "", inf)
    # Remove past plural ending (-un)
    root_past = re.sub(r"un$", "", past)

    # --- Remove 'w' ---
    # We remove 'w' from the stems entirely as it is not the locus of contrast
    root_inf = root_inf.replace("w", "")
    root_past = root_past.replace("w", "")

    # 3. Identify Locus of Allomorphy

    # A. Vowels (Ablaut)
    # Find the last vowel group in the root
    vowel_regex = r"([aeiouāēīōū]+)(?=[^aeiouāēīōū]*$)"

    v_inf_match = re.search(vowel_regex, root_inf)
    v_past_match = re.search(vowel_regex, root_past)

    has_ablaut = False
    if v_inf_match and v_past_match:
        if v_inf_match.group(1) != v_past_match.group(1):
            has_ablaut = True

    # B. Consonants (Verner's Law)

    # --- Find last consonant ---
    c_inf_match = re.search(r"([^aeiouāēīōū])$", root_inf)
    c_past_match = re.search(r"([^aeiouāēīōū])$", root_past)

    has_verner = False
    verner_pair = None

    # Valid Verner alternations (Voiceless -> Voiced)
    valid_alternations = [
        ("þ", "d"),
        ("f", "b"),
        ("f", "ƀ"),
        ("s", "z"),
        ("s", "r"),
        ("h", "g"),
        ("x", "g"),
    ]

    if c_inf_match and c_past_match:
        c1 = c_inf_match.group(1)
        c2 = c_past_match.group(1)

        if c1 != c2:
            for v_in, v_out in valid_alternations:
                # Direct comparison since we only have 1 character
                if c1 == v_in and c2 == v_out:
                    has_verner = True
                    verner_pair = f"{v_in}-{v_out}"
                    break

    return {
        "pgmc_inf": inf,
        "pgmc_past": past,
        "pgmc_root_inf": root_inf,
        "pgmc_root_past": root_past,
        "locus_vowel": has_ablaut,
        "locus_consonant": has_verner,
        "verner_pair": verner_pair,
    }


pgmc_data = dict()
for key in hg_to_proto_with_conj.keys():
    if key in mappable_lemmas:
        inf, past_pl = hg_to_proto_with_conj[key]
        pgmc_data_dict = analyze_pgmc_pair(inf, past_pl)
        pgmc_data[key] = pgmc_data_dict


for lemma in mappable_lemmas:
    try:
        lemma_ids = corpus_df[corpus_df["lemma"] == lemma]["lemma_id"]
        # if len(lemma_ids) == 0:
        #     continue
        lemma_id = lemma_ids.iloc[0]
        pgmc_data[lemma]["lemma_id"] = int(lemma_id)
    except:
        pgmc_data[lemma]["lemma_id"] = None


# Convert pgmc_data dict to DataFrame
pgmc_df = pd.DataFrame.from_dict(pgmc_data, orient="index")

# 1. Ensure we have a valid lemma_id (crucial for merging)
pgmc_df.dropna(subset=["lemma_id"], inplace=True)
pgmc_df["lemma_id"] = pgmc_df["lemma_id"].astype("int")


# 2. Define the types of allomorphy
def classify_allomorphy(row):
    # If it has consonant alternation (Verner) AND vowel alternation (Ablaut)
    if row["locus_consonant"] and row["locus_vowel"]:
        return "bipartite"
    # If it ONLY has vowel alternation (Standard Strong Verb without Verner)
    elif row["locus_vowel"] and not row["locus_consonant"]:
        return "non-bipartite"
    # Fallback/Edge cases (e.g. slight irregulars or scraping errors)
    else:
        return "other"


pgmc_df["allomorphy_type"] = pgmc_df.apply(classify_allomorphy, axis=1)

# 3. Filter the dataset to keep verbs that had some allomorphy
pgmc_df = pgmc_df[pgmc_df["allomorphy_type"].isin(["bipartite", "non-bipartite"])]

# Merge
df_merged = pd.merge(corpus_df, pgmc_df, on="lemma_id", how="left")


def standardize_infl(val):
    """
    Maps noisy inflection strings to Principal Parts logic.
    Returns: 'Pres', 'PastSg', 'PastPl', 'Ppl'
    """
    val = val.lower()

    # 1. Past Participle (PP4)
    if "participle" in val or "ppl" in val:
        return "Ppl"

    # 2. Past Tense
    if "past" in val or "prät" in val or "pret" in val:
        # Check for Plural OR 2nd Person Singular (often behaves like Plural in MHG)
        if "pl" in val or "2" in val:
            return "PastPl"  # Broadly PP3 context
        else:
            return "PastSg"  # Broadly PP2 context (1st/3rd Sg)

    # 3. Present / Infinitive (PP1)
    # Default bucket for everything else (Pres, Inf, Imp)
    return "Pres"


df_merged["std_infl"] = df_merged["infl"].map(standardize_infl)

### Code vowel and consonant alternation states...
