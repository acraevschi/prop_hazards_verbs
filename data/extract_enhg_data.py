import argparse
import os
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def extract_strong_verbs(folder_path="enhg_corpus", output_file="enhg_corpus.csv"):

    # if current directory doesn't contain `folder_path` and is not "data", change to it
    if not os.path.exists(folder_path):
        try:
            os.chdir("data")
            if not os.path.exists(folder_path):
                raise FileNotFoundError
        except FileNotFoundError:
            print(
                f"""
                Directory {folder_path} does not exist.\n
                Also tried: "data/{folder_path}"\n 
                Please check the path."""
            )
            return

    results = []
    lemma_inflClass_map = defaultdict(set)  # Store inflClass for lemmas

    # List subfolders to process
    subfolders = ["ref-mlu", "ref-rub"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # First pass: Process files and collect tokens
        for filename in tqdm(
            os.listdir(subfolder_path), desc=f"Processing {subfolder}"
        ):
            if not filename.endswith(".xml"):
                continue

            filepath = os.path.join(subfolder_path, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()

            # Parse header metadata
            metadata = {}
            header = root.find("header")
            if header is not None and header.text:
                lines = [
                    line.strip() for line in header.text.split("\n") if line.strip()
                ]
                for line in lines:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        metadata[key.strip()] = val.strip()

            ns = {"": ""}
            for token in root.findall(".//token", ns):
                token_id = token.get("id")
                tok_dipl = token.find("tok_dipl", ns)
                tok_anno = token.find("tok_anno", ns)

                if tok_anno is None:
                    continue

                # Check if verb (posLemma="VV")
                posLemma = tok_anno.find("posLemma", ns)
                if posLemma is None or posLemma.get("tag") != "VV":
                    continue

                pos = tok_anno.find("pos", ns)
                if pos is None:
                    continue
                pos_tag = pos.get("tag")

                # Initialize variables
                inflClass = ""
                infl = ""
                form = tok_dipl.get("utf", "") if tok_dipl is not None else ""
                norm = tok_anno.get("utf", "")
                lemma = (
                    tok_anno.find("lemma", ns).get("tag", "")
                    if tok_anno.find("lemma", ns) is not None
                    else ""
                )

                # Process verb types
                if pos_tag == "VVFIN":
                    morph = tok_anno.find("morph", ns)
                    if morph is None:
                        continue
                    morph_tag = morph.get("tag", "")
                    parts = morph_tag.split(".")
                    inflClass = parts[-1]
                    infl = ".".join(parts[:-1])
                    # Record inflClass for lemma
                    if inflClass and inflClass != "*":
                        lemma_inflClass_map[lemma].add(inflClass)

                elif pos_tag == "VVINF":
                    infl = "infinitive"
                    morph = tok_anno.find("morph", ns)
                    if morph is not None:
                        inflClass = morph.get("tag", "*").split(".")[-1]

                elif pos_tag == "VVPP":
                    infl = "past_participle"

                else:
                    continue

                # Append results with metadata
                results.append(
                    {
                        "form": form,
                        "norm": norm,
                        "lemma": lemma,
                        "inflClass": inflClass,
                        "grapho": "",
                        "infl": infl,
                        "language-region": metadata.get("language-region", ""),
                        "date": metadata.get("date", ""),
                        "time": metadata.get("time", ""),
                        "corpus": metadata.get("corpus", ""),
                        "id": token_id,
                        "specific_dating": "",
                        "lemma_count": 0,  # Placeholders for frequencies
                        "lemma_freq_per_1000": 0.0,
                        "form_count": 0,
                        "form_freq_per_1000": 0.0,
                    }
                )

    # Second pass: Update inflClass for infinitives/participles
    for token in results:
        if token["infl"] in ["infinitive", "past_participle"]:
            if (
                token["inflClass"] in ("", "*")
                and token["lemma"] in lemma_inflClass_map
            ):
                classes = lemma_inflClass_map[token["lemma"]]
                token["inflClass"] = "|".join(sorted(classes))

    # Compute frequencies
    total_tokens = len(results)
    lemma_counts = defaultdict(int)
    form_counts = defaultdict(int)

    for token in results:
        lemma_counts[token["lemma"]] += 1
        form_counts[token["norm"]] += 1

    for token in results:
        lemma = token["lemma"]
        form = token["norm"]
        token["lemma_count"] = lemma_counts[lemma]
        token["form_count"] = form_counts[form]
        if total_tokens > 0:
            token["lemma_freq_per_1000"] = (lemma_counts[lemma] / total_tokens) * 1000
            token["form_freq_per_1000"] = (form_counts[form] / total_tokens) * 1000

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ~results_df["inflClass"].isin(["Sw", "", "*", "Flekt", "Unflekt"])
    ]

    # save results to CSV
    print(f"Saving data to `{output_file}`...")
    results_df.to_csv(output_file, index=False)

    return results_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract strong verbs from the ENHG corpus."
    )

    parser.add_argument(
        "--folder_path",
        type=str,
        default="enhg_corpus",
        help="Path to the folder containing the ENHG corpus.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="enhg_corpus.csv",
        help="Path for the output CSV file.",
    )

    args = parser.parse_args()

    extract_strong_verbs(folder_path=args.folder_path, output_file=args.output_file)
