from tqdm import tqdm
import pandas as pd
import re
import json
from collections import defaultdict

mhg = pd.read_csv("./data/mhg_corpus.csv")
mhg_lemmas = mhg["lemma"].unique().tolist()

enhg = pd.read_csv("./data/enhg_corpus.csv")
enhg_lemmas = enhg["lemma"].unique().tolist()


def clean_text(text):
    # Remove parentheses for consistency
    return re.sub(r"[()]", "", text).strip()


mhg_clean_unclean = {}

for lemma in mhg_lemmas:
    # First, perform your common cleaning operation
    cleaned = clean_text(lemma)
    # Intermediate result after splitting on "/"
    unclean_slash = cleaned.split("/")[0]
    # Final cleaned lemma after splitting on "-"
    ultimate = unclean_slash.split("-")[-1]

    # Map both the original and the intermediate version to the final cleaned version
    mhg_clean_unclean[lemma] = ultimate
    mhg_clean_unclean[unclean_slash] = ultimate


with open("./data/lemmas/mhg_lemmas.txt", "w", encoding="utf-8") as f:
    for lemma in set(mhg_clean_unclean.values()):
        f.write(f"{lemma}\n")

with open("./data/lemmas/mhg_mapping.json", "w", encoding="utf-8") as f:
    json.dump(mhg_clean_unclean, f, ensure_ascii=False, indent=4)

### ENHG


def get_base_form(lemma, all_lemmas):
    if len(lemma) <= 4:
        return lemma
    # Find all lemmas that end with the same suffix and are substrings of the current lemma
    candidates = [l for l in all_lemmas if lemma.endswith(l) and l != lemma]
    if candidates:
        # Return the shortest candidate (most likely the base)
        return min(candidates, key=len)
    return lemma


temp_mapping = {}
all_ultimate1 = set()

for lemma in enhg_lemmas:
    # Skip invalid entries
    if isinstance(lemma, float) or not lemma:
        continue

    # Initial cleaning and splitting
    cleaned = clean_text(lemma)
    parts_slash = [part.strip() for part in cleaned.split("/") if part.strip()]
    if not parts_slash:
        continue
    unclean_slash = parts_slash[0]

    parts_dash = [part.strip() for part in unclean_slash.split("-") if part.strip()]
    if not parts_dash:
        continue
    ultimate1 = parts_dash[-1]

    # Store mappings
    temp_mapping[lemma] = ultimate1
    if unclean_slash != lemma:  # Avoid duplicate keys
        temp_mapping[unclean_slash] = ultimate1
    all_ultimate1.add(ultimate1)

# Generate base form mappings
base_form_map = {u: get_base_form(u, all_ultimate1) for u in all_ultimate1}

# Create final enhanced mapping
enhg_clean_unclean = {}
for key, u1 in temp_mapping.items():
    enhg_clean_unclean[key] = base_form_map[u1]

# Save processed data
with open("./data/lemmas/enhg_lemmas.txt", "w", encoding="utf-8") as f:
    for lemma in sorted(set(base_form_map.values())):
        f.write(f"{lemma}\n")

with open("./data/lemmas/enhg_mapping.json", "w", encoding="utf-8") as f:
    json.dump(enhg_clean_unclean, f, ensure_ascii=False, indent=4)


### Check linked list

mhg_enhg = pd.read_csv("./data/enhg_mhg_link.tsv", sep="\t")
mhg_enhg.fillna("", inplace=True)
mhg_enhg["doubt"] = mhg_enhg["ENHG"].apply(lambda x: "?" in x or "?" in x.split(","))
mhg_enhg["ENHG"] = mhg_enhg["ENHG"].apply(lambda x: x.replace("?", "").strip())
mhg_enhg["MHG"] = mhg_enhg["MHG"].apply(lambda x: x.strip())


mhg_enhg = mhg_enhg[mhg_enhg["MHG"].isin(mhg_lemmas_clean)]
mhg_lemmas_linked_lst = mhg_enhg[mhg_enhg["Comment"] == ""]["MHG"].tolist()

for lemma in mhg_lemmas_linked_lst:
    print(f"Linked MHG lemma: {lemma}")
