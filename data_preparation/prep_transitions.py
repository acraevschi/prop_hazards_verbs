import pandas as pd
import json
import os
import argparse


def prepare_transitions_data(data_path, output_path=None, covariates=["dialect_group"]):
    df = pd.read_csv(data_path, sep="\t", encoding="utf-8")

    with open("data/dialect_mapping.json", "r", encoding="utf-8") as f:
        dialect_mapping = json.load(f)

    df = df.dropna(subset=["dialect/place"])
    df["dialect_group"] = df["dialect/place"].map(dialect_mapping)
    df = df.dropna(subset=["dialect_group"])
    df = df[df["dialect_group"] != ""]

    df = df.drop_duplicates(subset=["lemma", "infl", "document"])

    # Create unique form_id from lemma and inflection
    df["form_id"] = df["lemma"].astype(str) + "_" + df["infl"].astype(str)

    # Sort data chronologically within each form_id group
    df = df.sort_values(["form_id", "date"])

    # Calculate time since previous observation
    df["prev_time"] = df.groupby("form_id")["date"].shift(1)
    df["time_since_prev"] = df["date"] - df["prev_time"]
    df["time_since_prev"] = df["time_since_prev"].fillna(0.0)

    # Convert coding columns to binary observations
    df["obs_v"] = (df["V.coding"] == "conservative").astype(int)
    df["obs_c"] = (df["C.coding"] == "conservative").astype(int)

    # Select and rename required columns
    output_cols = ["form_id", "date", "time_since_prev", "obs_v", "obs_c"] + covariates
    output_df = df[output_cols].rename(columns={"date": "time"})

    # Find form_ids that have at least two different time values
    time_counts = output_df.groupby("form_id")["time"].nunique()
    valid_forms = time_counts[time_counts >= 2].index

    # Filter to keep only rows with valid form_ids
    output_df = output_df[output_df["form_id"].isin(valid_forms)]
    output_df.reset_index(drop=True, inplace=True)

    if output_path:
        output_df.to_csv(output_path, index=False, encoding="utf-8")
        return
    else:
        return output_df  # if calling from another script


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
