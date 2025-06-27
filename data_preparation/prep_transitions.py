import pandas as pd
import json
import os
import argparse
import numpy as np


def prepare_transitions_data(data_path, output_path=None, covariates=["dialect_group"]):
    df = pd.read_csv(data_path, sep="\t", encoding="utf-8")

    with open("data/dialect_mapping.json", "r", encoding="utf-8") as f:
        dialect_mapping = json.load(f)

    df = df.dropna(subset=["dialect/place"])
    df["dialect_group"] = df["dialect/place"].map(dialect_mapping)
    df = df.dropna(subset=["dialect_group"])
    df = df[df["dialect_group"] != ""]

    df = df.drop_duplicates(subset=["lemma", "infl", "document"])

    # Map inflection to principal parts using both infl and POS columns
    df["principal_part_id"] = df.apply(
        lambda row: map_to_principal_part(row["infl"], row.get("POS", None)), axis=1
    )
    # Drop rows where principal_part_id is None
    df = df.dropna(subset=["principal_part_id"])
    # Make sure principal_part_id is integer
    df["principal_part_id"] = df["principal_part_id"].astype(int)

    # Create unique form_id from lemma and principal_part_id
    df["form_id"] = df["lemma"].astype(str) + "_" + df["principal_part_id"].astype(str)

    # Sort data chronologically within each form_id group
    df = df.sort_values(["form_id", "date"])

    # Calculate time since previous observation
    df["prev_time"] = df.groupby("form_id")["date"].shift(1)
    df["time_since_prev"] = df["date"] - df["prev_time"]
    df["time_since_prev"] = df["time_since_prev"].fillna(0.0)

    # Convert coding columns to binary observations
    df["obs_v"] = (df["V.coding"] == "conservative").astype(int)
    df["obs_c"] = (df["C.coding"] == "conservative").astype(int)

    # Create lemma mapping
    unique_lemmas = df["lemma"].unique()
    lemma_to_id = {lemma: i + 1 for i, lemma in enumerate(unique_lemmas)}
    df["lemma_id"] = df["lemma"].map(lemma_to_id)

    # Create list of unique time points
    unique_times = sorted(df["date"].unique())
    time_to_idx = {time: i + 1 for i, time in enumerate(unique_times)}

    irregularity_index, m_values = compute_irregularity_index(
        df, unique_lemmas, unique_times
    )

    # Select and rename required columns
    output_cols = [
        "form_id",
        "date",
        "time_since_prev",
        "obs_v",
        "obs_c",
        "lemma_id",
        "principal_part_id",
    ] + covariates
    output_df = df[output_cols].rename(columns={"date": "time"})

    # Find form_ids that have at least two different time values
    time_counts = output_df.groupby("form_id")["time"].nunique()
    valid_forms = time_counts[time_counts >= 2].index

    # Filter to keep only rows with valid form_ids
    output_df = output_df[output_df["form_id"].isin(valid_forms)]
    output_df.reset_index(drop=True, inplace=True)

    # Save the additional metadata (irregularity index, unique times, and m values)
    metadata = {
        "n_lemmas": int(len(unique_lemmas)),
        "n_principal_parts": int(df["principal_part_id"].nunique()),
        "n_time_points": int(len(unique_times)),
        "unique_times": [
            float(t) for t in unique_times
        ],  # Convert numpy values to Python float
        "irregularity_index": [
            [float(val) for val in row] for row in irregularity_index
        ],  # Convert 2D array to nested Python lists
        "m_values": [
            [int(val) for val in row] for row in m_values
        ],  # Add m-values to metadata
    }

    if output_path:
        output_df.to_csv(output_path, index=False, encoding="utf-8")
        # Save metadata to a separate file
        metadata_path = output_path.replace(".csv", "_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        return
    else:
        return output_df, metadata  # if calling from another script


def map_to_principal_part(infl, pos):
    """
    Map inflection and part-of-speech to principal part IDs.

    Args:
        infl: The inflection value
        pos: The part-of-speech tag

    Returns:
        int: Principal part ID (1-4) or None if cannot be determined
    """
    # If POS indicates a past participle, it's principal part 4
    if pos in ["VVPP", "VVPPA"]:
        return 4

    # If POS indicates an infinitive or imperative, it's principal part 1
    if pos in ["VVINF", "VVINFA", "VVIMP"]:
        return 1

    # If infl is missing or empty
    if pd.isna(infl) or infl in ["---", "--"]:
        return None

    # Principal part 1: Present system forms
    if any(x in str(infl) for x in ["Pres", "pres"]):
        return 1

    # Principal part 2: 1st/3rd singular past indicative
    if (
        (
            "Sg.1" in str(infl)
            or "Sg.3" in str(infl)
            or "Past.Sg.1" in str(infl)
            or "Past.Sg.3" in str(infl)
        )
        and ("Ind" in str(infl) or "ind" in str(infl))
        and ("Past" in str(infl) or "past" in str(infl))
    ):
        return 2

    # Principal part 3: Other past forms (plural past, 2sg past, past subjunctive)
    if "Past" in str(infl) or "past" in str(infl) or "Konj" in str(infl):
        return 3

    # For remaining forms that only have person/number marking without tense
    if any(x in str(infl) for x in ["Pl", "Sg"]):
        # separate category for forms with little info
        return None

    return None


def compute_irregularity_index(df, lemmas, time_points, alpha=0.5):
    """
    Compute irregularity index for each lemma at each time point based on
    the diversity of vowel and consonant observations across principal parts.

    Formula:
    - V_irreg = (|distinct vowels observed| - 1) / (m - 1)
    - C_irreg = (|distinct consonants observed| - 1) / (m - 1)
    - IrregIndex = alpha * V_irreg + (1 - alpha) * C_irreg

    where `m` is the number of principal parts observed.

    Args:
        df: DataFrame with preprocessed data
        lemmas: List of unique lemmas
        time_points: List of unique time points
        alpha: Weight for combining vowel and consonant irregularity (default 0.5)

    Returns:
        Tuple containing:
        - 2D list representing irregularity index for each lemma at each time point
        - 2D list representing number of principal parts (m) used for each calculation
    """
    n_lemmas = len(lemmas)
    n_times = len(time_points)

    # Initialize irregularity matrix and m-values matrix
    irregularity = np.zeros((n_lemmas, n_times))
    m_values = np.zeros((n_lemmas, n_times), dtype=int)

    lemma_to_idx = {lemma: i for i, lemma in enumerate(lemmas)}
    time_to_idx = {time: i for i, time in enumerate(time_points)}

    # Process each time point separately
    for time in time_points:
        time_idx = time_to_idx[time]
        time_data = df[df["date"] <= time]  # Use data up to this time point

        # Process each lemma
        for lemma in lemmas:
            lemma_idx = lemma_to_idx[lemma]
            lemma_data = time_data[time_data["lemma"] == lemma]

            if len(lemma_data) == 0:
                continue  # No data for this lemma at this time

            # Count principal parts observed
            observed_parts = lemma_data["principal_part_id"].nunique()

            # Store m value regardless of whether we can calculate irregularity
            m_values[lemma_idx, time_idx] = observed_parts

            if observed_parts <= 1:
                # Not enough principal parts to calculate irregularity
                irregularity[lemma_idx, time_idx] = 0.0
                continue

            # Count distinct vowel and consonant observations for each principal part
            vowel_obs = set()
            cons_obs = set()

            for _, part_data in lemma_data.groupby("principal_part_id"):
                # Get the most recent observation for each principal part
                latest = part_data.sort_values("date", ascending=False).iloc[0]
                vowel_obs.add(latest["V.obs"])
                cons_obs.add(latest["C.obs"])

            # Calculate irregularity components
            vowel_irreg = (len(vowel_obs) - 1) / (observed_parts - 1)
            cons_irreg = (len(cons_obs) - 1) / (observed_parts - 1)

            # Combined irregularity index
            irreg_index = alpha * vowel_irreg + (1 - alpha) * cons_irreg

            # Update irregularity score
            irregularity[lemma_idx, time_idx] = irreg_index

    # Forward fill missing values (time points with no data)
    for i in range(n_lemmas):
        last_valid_irreg = 0
        last_valid_m = 0
        for j in range(n_times):
            if irregularity[i, j] == 0 and m_values[i, j] == 0 and j > 0:
                irregularity[i, j] = last_valid_irreg
                m_values[i, j] = last_valid_m
            else:
                last_valid_irreg = irregularity[i, j]
                last_valid_m = m_values[i, j]

    return (
        irregularity.tolist(),  # Convert to list for JSON serialization
        m_values.tolist(),  # Convert m-values to list
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare transitions data from verb forms dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/verner_data_for_analysis.tsv",
        help="Path to input TSV data file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/transitions_data.csv",
        help="Path for output CSV file",
    )
    parser.add_argument(
        "--covariates",
        type=str,
        nargs="+",
        default=["dialect_group"],
        help="List of covariates to include in output (space-separated)",
    )

    args = parser.parse_args()

    # Check if data file exists in current directory, if not try parent
    if not os.path.exists(args.data_path):
        os.chdir("..")
        print(f"Changed to parent directory, looking for {args.data_path}")

    output = prepare_transitions_data(
        data_path=args.data_path,
        output_path=args.output_path,
        covariates=args.covariates,
    )

    print(f"Data processed and saved to {args.output_path}")
