import pandas as pd
import json

### TODO: Explain and make it more functional and readable
### At one point, will need to adapt to account for more covariates

### Irregularity coding

verner_df = pd.read_csv(
    "../data/verner_data_for_analysis.tsv", sep="\t", encoding="utf-8"
)

with open("../data/dialect_mapping.json", "r", encoding="utf-8") as f:
    dialect_mapping = json.load(f)

verner_df = verner_df.dropna(subset=["dialect/place"])

verner_df["dialect_group"] = verner_df["dialect/place"].map(dialect_mapping)

verner_df["dialect_group"].value_counts()  # too few Low German observations

verner_df = verner_df.groupby("dialect_group").filter(lambda x: len(x) >= 100)

verner_df["infl"] = verner_df["infl"].replace("--", "PastParticiple")

# Create unique unit identifier
verner_df["unit_id"] = (
    verner_df["dialect_group"] + "_" + verner_df["lemma"] + "_" + verner_df["infl"]
)

verner_df = verner_df.drop_duplicates(
    subset=["document", "unit_id", "C.coding", "V.coding"]
)

# Group observations by unit and date
grouped = (
    verner_df.groupby(["unit_id", "date"])
    .agg(
        {
            "C.coding": list,
            "V.coding": list,
            "dialect_group": "first",
            "lemma": "first",
            "infl": "first",
        }
    )
    .reset_index()
)


# Structure observations as time-series for each unit
def structure_observations(group):
    return sorted(
        [
            {"date": row["date"], "C": row["C.coding"], "V": row["V.coding"]}
            for _, row in group.iterrows()
        ],
        key=lambda x: x["date"],
    )


new_df = (
    grouped.groupby("unit_id")
    .apply(
        lambda x: pd.Series(
            {
                "dialect_group": x["dialect_group"].iloc[0],
                "lemma": x["lemma"].iloc[0],
                "infl": x["infl"].iloc[0],
                "observations": structure_observations(x),
            }
        )
    )
    .reset_index()
)

new_df["chain_length"] = new_df["observations"].apply(len)
new_df = new_df[new_df["chain_length"] >= 2]
new_df.reset_index(drop=True, inplace=True)

### JSON observations to transitions

# final_df.to_csv("verner_observations.csv", index=False, encoding="utf-8")
# Explode nested structures
exploded_df = new_df.explode("observations").reset_index(drop=True)
observations_df = pd.json_normalize(exploded_df["observations"])
processed_df = pd.concat([exploded_df[["unit_id"]], observations_df], axis=1)

# Handle multiple C/V pairs per observation
processed_df["C_V_pairs"] = processed_df.apply(
    lambda x: list(zip(x["C"], x["V"])), axis=1
)
final_exploded = processed_df.explode("C_V_pairs").reset_index(drop=True)
final_exploded[["C_obs", "V_obs"]] = pd.DataFrame(
    final_exploded["C_V_pairs"].tolist(), index=final_exploded.index
)

# Convert to binary values
final_exploded["obs_c"] = final_exploded["C_obs"].eq("conservative").astype(int)
final_exploded["obs_v"] = final_exploded["V_obs"].eq("conservative").astype(int)

# Create clean dataframe
clean_df = final_exploded[["unit_id", "date", "obs_c", "obs_v"]]
clean_df["date"] = clean_df["date"].astype(int)

clean_df = clean_df.sort_values(["unit_id", "date"])

# Calculate time differences using group-wise date progression
# Step 1: Get first occurrence dates for each verb
date_groups = clean_df.groupby(["unit_id", "date"]).size().reset_index().drop(0, axis=1)

# Step 2: Calculate time differences between consecutive date groups
date_groups["prev_date"] = date_groups.groupby("unit_id")["date"].shift(1)
date_groups["time_since_prev"] = date_groups["date"] - date_groups["prev_date"]
date_groups["time_since_prev"] = date_groups["time_since_prev"].fillna(0.0)

# Step 3: Merge back with original data
clean_df = clean_df.merge(
    date_groups[["unit_id", "date", "time_since_prev"]],
    on=["unit_id", "date"],
    how="left",
)

# Create final output
output_df = clean_df.rename(columns={"unit_id": "verb", "date": "time"})[
    ["verb", "time", "time_since_prev", "obs_v", "obs_c"]
]

output_df.to_csv("../data/verner_transitions.csv", index=False)
