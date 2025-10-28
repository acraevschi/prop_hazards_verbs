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
                    "modern_lemma_count": row.get("modern_lemma_count", np.nan),
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


def prepare_aggregated_markov_data(df, date_col="date"):
    """
    Input df: token-level rows with at least the columns:
      lemma_id, principal_part, date, variety, corpus,
      vowel_alternation, cons_alternation,
      form_freq_per_1000, lemma_freq_per_1000

    Output: obs_df (aggregated rows: one row per lemma x part x date),
            seq_df (sequence metadata; one row per lemma x part),
            variety_map, corpus_map
    """

    df = df.copy()

    # 1) map states
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

    df["state"] = df.apply(
        lambda x: map_state(x["vowel_alternation"], x["cons_alternation"]), axis=1
    )

    # 2) aggregate to lemma x principal_part x date
    # compute counts per state, totals, and average predictors
    agg = (
        df.groupby(["lemma_id", "principal_part", date_col])
        .agg(
            n_total=("state", "size"),
            n1=("state", lambda s: (s == 1).sum()),
            n2=("state", lambda s: (s == 2).sum()),
            n3=("state", lambda s: (s == 3).sum()),
            n4=("state", lambda s: (s == 4).sum()),
            avg_form_freq=("form_freq_per_1000", "mean"),
            avg_lemma_freq=("lemma_freq_per_1000", "mean"),
            modern_lemma_count=("modern_lemma_count", "mean"),
            # pick modal variety/corpus within this cell (use first if tie)
            variety_mode=(
                "variety",
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan,
            ),
            corpus_mode=(
                "corpus",
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan,
            ),
        )
        .reset_index()
    )

    # 3) lemma-level bipartite proportion per date
    lemma_bip = (
        df.assign(is_bipartite=(df["state"] == 4).astype(int))
        .groupby(["lemma_id", date_col])["is_bipartite"]
        .agg(n_bipartite="sum", n_total="count")
        .reset_index()
    )
    lemma_bip["prop_bipartite"] = lemma_bip["n_bipartite"] / lemma_bip["n_total"]

    # merge lemma-level bipartite proportion into aggregated rows
    agg = agg.merge(
        lemma_bip[["lemma_id", date_col, "prop_bipartite"]],
        on=["lemma_id", date_col],
        how="left",
    )
    agg["prop_bipartite"] = agg["prop_bipartite"].fillna(0.0)

    # 4) build sequences per (lemma_id, principal_part)
    sequences = []
    rows = []
    seq_id = 0

    # sort globally for stable ordering
    agg = agg.sort_values(["lemma_id", "principal_part", date_col]).reset_index(
        drop=True
    )

    for (lemma_id, principal_part), group in agg.groupby(
        ["lemma_id", "principal_part"]
    ):
        group = group.sort_values(date_col).reset_index(drop=True)

        # require at least 2 aggregated timepoints for a transition to occur
        if len(group) < 2:
            continue

        # we'll create a sequence even if length 1 (useful)
        seq_id += 1
        for i, row in group.iterrows():
            rows.append(
                {
                    "seq_id": seq_id,
                    "lemma_id": lemma_id,
                    "principal_part": principal_part,
                    "obs_index": i + 1,
                    "date": row[date_col],
                    "n1": int(row["n1"]),
                    "n2": int(row["n2"]),
                    "n3": int(row["n3"]),
                    "n4": int(row["n4"]),
                    "n_total": int(row["n_total"]),
                    "avg_form_freq": row["avg_form_freq"],
                    "avg_lemma_freq": row["avg_lemma_freq"],
                    "prop_bipartite": row["prop_bipartite"],
                    "variety": row["variety_mode"],
                    "corpus": row["corpus_mode"],
                    "modern_lemma_count": row.get("modern_lemma_count", np.nan),
                }
            )

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

    if obs_df.empty:
        return obs_df, seq_df, {}, {}

    # reindex lemma_id to contiguous integers for Stan
    unique_lemmas = sorted(obs_df["lemma_id"].unique())
    lemma_id_mapping = {old: new for new, old in enumerate(unique_lemmas, start=1)}
    obs_df["lemma_id"] = obs_df["lemma_id"].map(lemma_id_mapping)
    seq_df["lemma_id"] = seq_df["lemma_id"].map(lemma_id_mapping)

    # map variety & corpus to integers (use modes present)
    varieties = [v for v in obs_df["variety"].dropna().unique()]
    variety_map = {v: i + 1 for i, v in enumerate(varieties)}
    corpora = [c for c in obs_df["corpus"].dropna().unique()]
    corpus_map = {c: i + 1 for i, c in enumerate(corpora)}

    obs_df["variety_code"] = obs_df["variety"].map(variety_map)
    obs_df["corpus_code"] = obs_df["corpus"].map(corpus_map)

    # 5) compute time intervals (year differences) between aggregated rows in each sequence
    obs_df = obs_df.sort_values(["seq_id", "obs_index"]).reset_index(drop=True)
    obs_df["time_interval_to_next"] = np.nan
    for sid, group in obs_df.groupby("seq_id"):
        idx = group.index.values
        dates = group["date"].values.astype(float)
        if len(dates) >= 2:
            deltas = np.diff(dates)
            obs_df.loc[idx[:-1], "time_interval_to_next"] = deltas
        obs_df.loc[idx[-1], "time_interval_to_next"] = np.nan

    # replace NaN times with 0.0 for storage convenience
    obs_df["time_interval_to_next"] = obs_df["time_interval_to_next"].fillna(0.0)

    return obs_df, seq_df, variety_map, corpus_map


def prepare_aggregated_markov_data_variety(df, date_col="date"):
    """
    Input df: token-level rows with at least the columns:
      lemma_id, principal_part, date, variety, corpus,
      vowel_alternation, cons_alternation,
      form_freq_per_1000, lemma_freq_per_1000

    Output: obs_df (aggregated rows: one row per lemma x part x date),
            seq_df (sequence metadata; one row per lemma x part),
            variety_map, corpus_map
    """

    df = df.copy()

    # 1) map states (can be done once on the whole dataframe)
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

    df["state"] = df.apply(
        lambda x: map_state(x["vowel_alternation"], x["cons_alternation"]), axis=1
    )

    all_obs = []
    all_seqs = []
    global_seq_id = 0

    # Process each variety separately
    for variety_name, variety_df in df.groupby("variety"):
        # 2) aggregate to lemma x principal_part x date within the variety
        agg = (
            variety_df.groupby(["lemma_id", "principal_part", date_col])
            .agg(
                n_total=("state", "size"),
                n1=("state", lambda s: (s == 1).sum()),
                n2=("state", lambda s: (s == 2).sum()),
                n3=("state", lambda s: (s == 3).sum()),
                n4=("state", lambda s: (s == 4).sum()),
                avg_form_freq=("form_freq_per_1000", "mean"),
                avg_lemma_freq=("lemma_freq_per_1000", "mean"),
                modern_lemma_count=("modern_lemma_count", "mean"),
                corpus_mode=(
                    "corpus",
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan,
                ),
            )
            .reset_index()
        )

        # 3) lemma-level bipartite proportion per date within the variety
        lemma_bip = (
            variety_df.assign(is_bipartite=(variety_df["state"] == 4).astype(int))
            .groupby(["lemma_id", date_col])["is_bipartite"]
            .agg(n_bipartite="sum", n_total="count")
            .reset_index()
        )
        lemma_bip["prop_bipartite"] = lemma_bip["n_bipartite"] / lemma_bip["n_total"]

        agg = agg.merge(
            lemma_bip[["lemma_id", date_col, "prop_bipartite"]],
            on=["lemma_id", date_col],
            how="left",
        )
        agg["prop_bipartite"] = agg["prop_bipartite"].fillna(0.0)

        # 4) build sequences per (lemma_id, principal_part) within the variety
        sequences = []
        rows = []

        agg = agg.sort_values(["lemma_id", "principal_part", date_col]).reset_index(
            drop=True
        )

        for (lemma_id, principal_part), group in agg.groupby(
            ["lemma_id", "principal_part"]
        ):
            group = group.sort_values(date_col).reset_index(drop=True)

            if len(group) < 2:
                continue

            global_seq_id += 1
            for i, row in group.iterrows():
                rows.append(
                    {
                        "seq_id": global_seq_id,
                        "lemma_id": lemma_id,
                        "principal_part": principal_part,
                        "obs_index": i + 1,
                        "date": row[date_col],
                        "n1": int(row["n1"]),
                        "n2": int(row["n2"]),
                        "n3": int(row["n3"]),
                        "n4": int(row["n4"]),
                        "n_total": int(row["n_total"]),
                        "avg_form_freq": row["avg_form_freq"],
                        "avg_lemma_freq": row["avg_lemma_freq"],
                        "prop_bipartite": row["prop_bipartite"],
                        "variety": variety_name,
                        "corpus": row["corpus_mode"],
                        "modern_lemma_count": row.get("modern_lemma_count", np.nan),
                    }
                )

            sequences.append(
                {
                    "seq_id": global_seq_id,
                    "lemma_id": lemma_id,
                    "principal_part": principal_part,
                    "n_obs": len(group),
                }
            )

        if rows:
            all_obs.append(pd.DataFrame(rows))
        if sequences:
            all_seqs.append(pd.DataFrame(sequences))

    # Concatenate results from all varieties
    if not all_obs:
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    obs_df = pd.concat(all_obs, ignore_index=True)
    seq_df = pd.concat(all_seqs, ignore_index=True)

    # The rest of the processing happens on the combined dataframe
    # to ensure consistent mappings and calculations.

    # reindex lemma_id to contiguous integers for Stan
    unique_lemmas = sorted(obs_df["lemma_id"].unique())
    lemma_id_mapping = {old: new for new, old in enumerate(unique_lemmas, start=1)}
    obs_df["lemma_id"] = obs_df["lemma_id"].map(lemma_id_mapping)
    seq_df["lemma_id"] = seq_df["lemma_id"].map(lemma_id_mapping)

    # map variety & corpus to integers (use modes present)
    varieties = sorted([v for v in obs_df["variety"].dropna().unique()])
    variety_map = {v: i + 1 for i, v in enumerate(varieties)}
    corpora = sorted([c for c in obs_df["corpus"].dropna().unique()])
    corpus_map = {c: i + 1 for i, c in enumerate(corpora)}

    obs_df["variety_code"] = obs_df["variety"].map(variety_map)
    obs_df["corpus_code"] = obs_df["corpus"].map(corpus_map)

    # 5) compute time intervals (year differences) between aggregated rows in each sequence
    obs_df = obs_df.sort_values(["seq_id", "obs_index"]).reset_index(drop=True)
    obs_df["time_interval_to_next"] = np.nan
    for sid, group in obs_df.groupby("seq_id"):
        idx = group.index.values
        dates = group["date"].values.astype(float)
        if len(dates) >= 2:
            deltas = np.diff(dates)
            obs_df.loc[idx[:-1], "time_interval_to_next"] = deltas
        obs_df.loc[idx[-1], "time_interval_to_next"] = np.nan

    # replace NaN times with 0.0 for storage convenience
    obs_df["time_interval_to_next"] = obs_df["time_interval_to_next"].fillna(0.0)

    return obs_df, seq_df, variety_map, corpus_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_obs", required=True)
    parser.add_argument("--out_seq", required=True)
    parser.add_argument("--aggregated", action="store_true", default=False)
    parser.add_argument("--by_variety", action="store_true", default=False)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.aggregated and not args.by_variety:
        obs_df, seq_df, variety_map, corpus_map = prepare_aggregated_markov_data(df)
    elif args.aggregated and args.by_variety:
        obs_df, seq_df, variety_map, corpus_map = (
            prepare_aggregated_markov_data_variety(df)
        )
    else:
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
