import argparse
import json
import os
from tqdm import tqdm
import pandas as pd
from collections import Counter


def extract_strong_verbs(folder_path="mhg_corpus", output_file="mhg_corpus.csv"):

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

    # Get all files in the corpus directory
    filenames = os.listdir(folder_path)
    tokens_to_keep = ["form", "norm", "lemma", "inflClass", "grapho", "infl"]
    metadata_to_keep = [
        "language-region",
        "date",
        "time",
        "id",  # id for document id
        "specific_dating",  # not always available
    ]

    # List to store all tokens with their metadata
    all_tokens = []
    total_word_count = 0

    print("Processing corpus files...")
    for filename in tqdm(filenames):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract metadata for this document
        metadata = {}
        if "metadata" in data:
            for field in metadata_to_keep:
                if field in data["metadata"]:
                    metadata[field] = data["metadata"][field]
                else:
                    metadata[field] = None

        # Process tokens
        if "token" in data:
            # Count total words for frequency calculation
            total_word_count += len(data["token"])

            for token in data["token"]:
                # skip empty/strange tokens
                if not token.get("form") or token.get("form") == "--":
                    continue

                # only verbs
                if not token.get("pos_hits").startswith("VV"):
                    continue

                # Skip tokens with inflClass = "v"
                if token.get("inflClass") == "v":
                    continue

                # Extract only the fields we want to keep
                token_data = {}
                for field in tokens_to_keep:
                    if field in token:
                        token_data[field] = token[field]
                    else:
                        token_data[field] = None

                # Special treatment for infinitives and past participles
                if token.get("pos_hits") == "VVINF":
                    token_data["infl"] = "infinitive"
                elif token.get("pos_hits") == "VVPP":
                    token_data["infl"] = "past_participle"

                # Add metadata to token
                token_data.update(metadata)
                all_tokens.append(token_data)

    # Convert to DataFrame
    df = pd.DataFrame(all_tokens)

    # Calculate lemma and token frequencies
    lemma_counts = Counter(df["lemma"].dropna())
    form_counts = Counter(df["form"].dropna())

    # Create dictionaries for frequency lookup
    lemma_freq_dict = {
        lemma: {
            "lemma_count": count,
            "lemma_freq_per_1000": count * 1000 / total_word_count,
        }
        for lemma, count in lemma_counts.items()
    }

    form_freq_dict = {
        form: {
            "form_count": count,
            "form_freq_per_1000": count * 1000 / total_word_count,
        }
        for form, count in form_counts.items()
    }

    # Add frequency information to the main DataFrame
    df["lemma_count"] = df["lemma"].map(
        lambda x: lemma_freq_dict.get(x, {}).get("lemma_count", 0)
    )
    df["lemma_freq_per_1000"] = df["lemma"].map(
        lambda x: lemma_freq_dict.get(x, {}).get("lemma_freq_per_1000", 0)
    )
    df["form_count"] = df["form"].map(
        lambda x: form_freq_dict.get(x, {}).get("form_count", 0)
    )
    df["form_freq_per_1000"] = df["form"].map(
        lambda x: form_freq_dict.get(x, {}).get("form_freq_per_1000", 0)
    )

    # add column "corpus" that is equal to "MHG_Referenzkorpus"
    df["corpus"] = "MHG_Referenzkorpus"

    # exclude weak verbs
    df = df[df["inflClass"] != "wk"]

    # Save the combined dataset
    print(f"Saving data to `{output_file}`...")
    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract corpus data from the MHG corpus."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default="mhg_corpus",
        help="Path to the folder containing the MHG corpus.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="mhg_corpus.csv",
        help="Path for the output CSV file for the extracted corpus data.",
    )
    args = parser.parse_args()

    extract_strong_verbs(folder_path=args.folder_path, output_file=args.output_file)
