from tqdm import tqdm
import pandas as pd
import re
import json
import requests
from lxml import html
import time

# ----------------------------
# Basic cleaning helpers
# ----------------------------


def clean_for_query(lemma: str) -> str:
    """
    Clean lemma for querying DWDS:
    - Remove parentheses
    - Split on '/' and take first
    - Split on '-' and take last
    - Split on '|' and take second part if available
    - Strip whitespace
    """
    if not lemma or not isinstance(lemma, str):
        return ""
    # Remove parentheses
    text = re.sub(r"[()]", "", lemma).strip()
    # Split on '/'
    text = text.split("/")[0].strip()
    # Split on '-'
    text = text.split("-")[-1].strip()
    # Split on '|'
    parts = [p.strip() for p in text.split("|") if p.strip()]
    if len(parts) > 1:
        text = parts[1]  # take second part if available
    elif parts:
        text = parts[0]
    return text


def looks_like_single_token(lemma: str) -> bool:
    """Reject multiword tokens and weird stuff."""
    if not lemma:
        return False
    if re.search(r"\s|[\/,;:()=\[\]{}<>\"']", lemma):
        return False
    if len(lemma) < 2:
        return False
    if re.search(r"\d", lemma):
        return False
    return True


# ----------------------------
# Load ENHG corpus
# ----------------------------
enhg = pd.read_csv("./data/enhg_corpus.csv", dtype=str)
enhg_lemmas = enhg["lemma"].dropna().unique().tolist()

# ----------------------------
# DWDS config
# ----------------------------
DWDS_BASE = "https://www.dwds.de/wb/etymwb"
XPATH_HEAD = '//*[@id="wb-1"]/div[3]/span[1]'

session = requests.Session()
session.headers.update({"User-Agent": "MHG-ENHG-Mapping (alexandru.craevschi@uzh.ch)"})

out_path = "./data/lemmas/enhg_mapping.json"

results = {}
failed = {}

# ----------------------------
# Process ENHG lemmas
# ----------------------------
for lemma in tqdm(enhg_lemmas, desc="DWDS lookups"):
    if lemma in results:
        continue

    cleaned = clean_for_query(lemma)

    if not looks_like_single_token(cleaned):
        continue

    url = f"{DWDS_BASE}/{cleaned}"
    try:
        resp = session.get(url, timeout=12)
        resp.raise_for_status()
    except requests.RequestException as e:
        failed[lemma] = f"request_failed: {e}"
        time.sleep(0.5)
        continue

    try:
        tree = html.fromstring(resp.content)
        nodes = tree.xpath(XPATH_HEAD)
        if not nodes:
            failed[lemma] = "no_headword_node"
            time.sleep(0.2)
            continue

        head_node = nodes[0]
        orth_nodes = head_node.xpath('.//span[contains(@class,"etymwb-orth")]')
        gram_nodes = head_node.xpath('.//span[contains(@class,"etymwb-gramgrp")]')

        if not orth_nodes:
            failed[lemma] = "no_orth"
            time.sleep(0.2)
            continue

        headword = orth_nodes[0].text_content().strip()
        pos_text = gram_nodes[0].text_content().strip() if gram_nodes else ""
        pos_norm = re.sub(r"[^\w]", "", pos_text)

        # Only keep verbs
        if "Vb" not in pos_norm and "vb" not in pos_norm and pos_norm != "V":
            failed[lemma] = f"not_verb: {pos_text}"
            time.sleep(0.15)
            continue

        # Clean headword: remove any digits and strip all whitespace
        headword = re.sub(r"\d+", "", headword)
        headword = re.sub(r"\s+", "", headword)

        # Map original lemma (uncleaned) -> base headword
        results[lemma] = headword

        time.sleep(0.15)

    except Exception as e:
        failed[lemma] = f"parse_error: {e}"
        time.sleep(0.2)
        continue

# Save results
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(
    f"DWDS lookup completed. Successful mappings: {len(results)}. Failures: {len(failed)}"
)
