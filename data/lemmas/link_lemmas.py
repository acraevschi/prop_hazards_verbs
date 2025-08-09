import requests
from lxml import html
import re
from tqdm import tqdm
import csv
import pandas as pd
import json

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SERVER_SEED = 97 # not used here but rather when starting LM Studio server
LM_STUDIO_URL = "http://localhost:1234/v1/completions"   # LM Studio endpoint
DWDS_BASE_URL  = "https://www.dwds.de/wb/etymwb"
X_PATH         = "/html/body/main/div[1]/div/div[1]/div[1]/div[3]"
HEADERS        = {"Content-Type": "application/json"}

# ------------------------------------------------------------------
# Helper: scrape the etymology section from DWDS
# ------------------------------------------------------------------
def get_etymology_section(lemma: str) -> str:
    """
    Returns only the text of the “Etymologie” block (no tags).
    Empty string on failure.
    """
    try:
        response = requests.get(f"{DWDS_BASE_URL}/{lemma}", timeout=10)
        response.raise_for_status()
        tree   = html.fromstring(response.content)
        elem   = tree.xpath(X_PATH)

        if not elem:
            return ""

        # Return all descendant text (no HTML tags)
        return elem[0].text_content().strip()

    except Exception as e:
        print(f"Error fetching {lemma}: {e}")
        return ""

# ------------------------------------------------------------------
# Helper: extract MHG forms from an etymology string
# ------------------------------------------------------------------
def extract_mhg_forms(etym_text: str) -> list[str]:
    """
    Find every word that follows the token “mhd.”.
    Example:  '… mhd. springen …' → ['springen', ...]
    """
    # Capture the word after "mhd." up to whitespace, comma or semicolon
    return re.findall(r"mhd\.\s+([^\s,;]+)", etym_text)

# ------------------------------------------------------------------
# Helper: Levenshtein distance (simple implementation)
# ------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    """
    Compute the edit distance between two strings.
    """
    if len(a) < len(b):
        return levenshtein(b, a)

    # Now a is longer or equal
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current_row = [i]
        for j, cb in enumerate(b, 1):
            insertions   = previous_row[j] + 1
            deletions    = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# ------------------------------------------------------------------
# Helper: ask the LLM for the best matching MHG lemmas
# ------------------------------------------------------------------
def query_llm(enhg_lemma: str,
              etymology_text: str,
              mhg_candidates: list[str]) -> str:
    """
    Returns raw text from LM Studio. The prompt asks the model to output
    a comma‑separated list of 1–3 best candidates inside square brackets.
    Example answer: "[springen, bespringen]"
    """

    # If we have no candidates, just return an empty list
    if not mhg_candidates:
        return "[]"

    prompt = f"""
Analyze the following Early New High German (ENHG) lemma and its etymological data to identify the most likely Middle High German (MHG) lemma(s) from the provided candidate list.

ENHG Lemma: {enhg_lemma}
Etymology Text: {etymology_text}

Candidate MHG Lemmas: {', '.join(mhg_candidates)}

Please output a list of 1–3 MHG lemmas that you think are most likely linked to the ENHG lemma. Output should be in square brackets, comma‑separated, e.g., "[first_match, second_match]". If only one candidate is certain, output just that single form inside brackets.
"""

    payload = {
        "prompt": prompt,
        "max_tokens": -1,
        "temperature": 0.3
    }

    try:
        r = requests.post(LM_STUDIO_URL,
                          json=payload,
                          headers=HEADERS,
                          timeout=120)
        return r.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"LLM query failed: {e}")
        # Safe fallback – empty list
        return "[]"

# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def process_lemmas(enhg_lemmas: list[str],
                   mhg_candidates: list[str],
                   output_file: str) -> None:
    """
    Writes a CSV with columns:
        ENHG Lemma, MHG Candidates, Link
    """

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["ENHG Lemma", "MHG Candidates", "Link"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for lemma in tqdm(enhg_lemmas, desc="Processing lemmas"):
            # 1️⃣ Scrape etymology
            etymology_text = get_etymology_section(lemma)
            link           = f"{DWDS_BASE_URL}/{lemma}"

            if not etymology_text:
                writer.writerow({
                    "ENHG Lemma": lemma,
                    "MHG Candidates": "",
                    "Link": link
                })
                continue

            # 2️⃣ Extract MHG forms from the text
            extracted_forms = extract_mhg_forms(etymology_text)

            if not extracted_forms:
                writer.writerow({
                    "ENHG Lemma": lemma,
                    "MHG Candidates": "",
                    "Link": link
                })
                continue

            # 3️⃣ Build a filtered candidate list (exact matches + top‑30 by edit distance)
            # Exact matches are automatically included because they have distance 0.
            candidate_set = set()

            for form in extracted_forms:
                distances = [(cand, levenshtein(form, cand)) for cand in mhg_candidates]
                distances.sort(key=lambda x: x[1])
                top_cands = [cand for cand, _ in distances[:30]]
                candidate_set.update(top_cands)

            filtered_candidates = sorted(candidate_set)  # deterministic order

            # 4️⃣ Ask the LLM
            llm_resp = query_llm(lemma, etymology_text, filtered_candidates)

            # 5️⃣ Parse the LLM output (e.g. "[springen, bespringen]")
            matches = re.findall(r'\[([^\]]*)\]', llm_resp)
            if matches:
                content = matches[-1]  # Take last match (most likely the output)
                # Split by comma and clean each candidate
                selected_cands = [c.strip() for c in content.split(",") if c.strip()]
            else:
                selected_cands = []
            # Keep only the first 3 (just in case the model returned more)
            selected_cands = selected_cands[:3]
            selected_str = "; ".join(selected_cands)

            writer.writerow({
                "ENHG Lemma": lemma,
                "MHG Candidates": selected_str,
                "Link": link
            })

            print(f"Processed: {lemma} | Selected: {selected_str}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    enhg_df   = pd.read_csv("./data/enhg_corpus.csv")
    input_enhg = enhg_df["lemma"].unique().tolist()

    mhg_mapping     = json.load(open("./data/lemmas/mhg_mapping.json",
                                     encoding="utf-8"))
    mhg_candidates  = list(set(mhg_mapping.values()))

    process_lemmas(input_enhg,
                   mhg_candidates,
                   output_file="./data/lemmas/etymology_matches.csv")
