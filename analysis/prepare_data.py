import pandas as pd
import numpy as np
import argparse


def map_state(vowel_alt, cons_alt):
    if vowel_alt == "no" and cons_alt == "no":
        return 1
    elif vowel_alt == "yes" and cons_alt == "no":
        return 2
    elif vowel_alt == "no" and cons_alt == "yes":
        return 3
    elif vowel_alt == "yes" and cons_alt == "yes":
        return 4
    else:
        return np.nan


def prepare_markov_data(df):
    df = df.copy()
    df["state"] = df.apply(
        lambda x: map_state(x["vowel_alternation"], x["cons_alternation"]), axis=1
    )

    # Sort by lemma_id, principal_part, date
    df = df.sort_values(["lemma_id", "principal_part", "date"]).reset_index(drop=True)

    # Maps
    varieties = df["variety"].unique()
    variety_map = {v: i + 1 for i, v in enumerate(varieties)}
    corpora = df["corpus"].unique()
    corpus_map = {c: i + 1 for i, c in enumerate(corpora)}

    # Average form frequency per lemma x principal part (used as predictor per observation)
    form_freq_avg = (
        df.groupby(["lemma_id", "principal_part"])["form_freq_per_1000"]
        .mean()
        .reset_index()
    )
    form_freq_avg = form_freq_avg.rename(
        columns={"form_freq_per_1000": "avg_form_freq"}
    )
    df = df.merge(form_freq_avg, on=["lemma_id", "principal_part"], how="left")

    # Build sequences per (lemma_id, principal_part)
    sequences = []
    rows = []
    seq_id = 0

    for (lemma_id, principal_part), group in df.groupby(["lemma_id", "principal_part"]):
        group = group.sort_values("date").reset_index(drop=True)

        # Filter out groups with fewer than 2 observations
        if len(group) < 2:
            continue

        seq_id += 1
        for i in range(len(group)):
            row = group.iloc[i]
            rows.append(
                {
                    "seq_id": seq_id,
                    "lemma_id": lemma_id,
                    "principal_part": principal_part,
                    "obs_index": i + 1,
                    "date": row["date"],
                    "state": row["state"],
                    "form_freq": row.get("avg_form_freq", np.nan),
                    "lemma_freq": row.get("lemma_freq_per_1000", np.nan),
                    "variety": variety_map[row["variety"]],
                    "corpus": corpus_map[row["corpus"]],
                }
            )

        # store metadata for sequence
        sequences.append(
            {
                "seq_id": seq_id,
                "lemma_id": lemma_id,
                "principal_part": principal_part,
                "n_obs": len(group),
            }
        )

    obs_df = pd.DataFrame(rows)
    seq_df = pd.DataFrame(sequences)

    # If no sequences were created, return empty dataframes
    if len(obs_df) == 0:
        return obs_df, seq_df, variety_map, corpus_map

    # Create a mapping from old lemma_id to new contiguous lemma_id
    unique_lemma_ids = sorted(obs_df["lemma_id"].unique())
    lemma_id_mapping = {
        old_id: new_id for new_id, old_id in enumerate(unique_lemma_ids, 1)
    }

    # Apply the mapping to both dataframes
    obs_df["lemma_id"] = obs_df["lemma_id"].map(lemma_id_mapping)
    seq_df["lemma_id"] = seq_df["lemma_id"].map(lemma_id_mapping)

    # Calculate time intervals between consecutive observations per sequence
    obs_df = obs_df.sort_values(["seq_id", "obs_index"]).reset_index(drop=True)
    obs_df["time_interval_to_next"] = np.nan

    for sid, g in obs_df.groupby("seq_id"):
        idx = g.index.values
        dates = g["date"].values
        if len(dates) >= 2:
            deltas = np.diff(dates)
            obs_df.loc[idx[:-1], "time_interval_to_next"] = deltas
        # last obs has no next interval; set to NaN or 0
        obs_df.loc[idx[-1], "time_interval_to_next"] = np.nan

    return obs_df, seq_df, variety_map, corpus_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_obs", required=True)
    parser.add_argument("--out_seq", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    obs_df, seq_df, variety_map, corpus_map = prepare_markov_data(df)

    obs_df.to_csv(args.out_obs, index=False)
    seq_df.to_csv(args.out_seq, index=False)

    pd.DataFrame(list(variety_map.items()), columns=["variety", "code"]).to_csv(
        "variety_mapping.csv", index=False
    )
    pd.DataFrame(list(corpus_map.items()), columns=["corpus", "code"]).to_csv(
        "corpus_mapping.csv", index=False
    )

    print(f"Wrote {len(obs_df)} observations in {len(seq_df)} sequences.")
    print(f"Unique lemma IDs in obs_df: {obs_df['lemma_id'].nunique()}")
    print(
        f"Max lemma ID in obs_df: {obs_df['lemma_id'].max() if len(obs_df) > 0 else 'N/A'}"
    )
    print(
        f"Lemma ID range: 1 to {obs_df['lemma_id'].max() if len(obs_df) > 0 else 'N/A'}"
    )


if __name__ == "__main__":
    main()
