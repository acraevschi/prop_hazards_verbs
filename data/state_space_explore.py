import pandas as pd
import json

data = pd.read_csv("data/verner_data_for_analysis.tsv", sep="\t", encoding="utf-8")
dialect_mapping = json.load(open("data/dialect_mapping.json", "r", encoding="utf-8"))

data["dialect"] = data["dialect/place"].map(dialect_mapping)

C_V_coding = []

for i, row in data.iterrows():
    if not row["C.coding"] or not row["V.coding"]:
        C_V_coding.append(None)
    elif row["C.coding"] == "conservative" and row["V.coding"] == "conservative":
        C_V_coding.append("conservative")
    elif row["C.coding"] == "conservative" and row["V.coding"] == "innovative":
        C_V_coding.append("V_innovative")
    elif row["C.coding"] == "innovative" and row["V.coding"] == "conservative":
        C_V_coding.append("C_innovative")
    elif row["C.coding"] == "innovative" and row["V.coding"] == "innovative":
        C_V_coding.append("innovative")

data["C_V_coding"] = C_V_coding


def map_to_principal_part(infl, pos):
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
    if any(x in infl for x in ["Pres", "pres"]):
        return 1

    # Principal part 2: 1st/3rd singular past indicative
    if (
        ("Sg.1" in infl or "Sg.3" in infl or "Past.Sg.1" in infl or "Past.Sg.3" in infl)
        and ("Ind" in infl or "ind" in infl)
        and ("Past" in infl or "past" in infl)
    ):
        return 2

    # Principal part 3: Other past forms (plural past, 2sg past, past subjunctive)
    if "Past" in infl or "past" in infl or "Konj" in infl:
        return 3

    # For remaining forms that only have person/number marking without tense
    if any(x in infl for x in ["Pl", "Sg"]):
        # separate category for forms with little info
        return 5

    return None


# Apply the function to create the principal_part column
data["principal_part"] = data.apply(
    lambda x: map_to_principal_part(x["infl"], x["POS"]), axis=1
)

# drop NA values from the relevant columns
data_drop = data.dropna(subset=["date", "dialect", "principal_part", "C_V_coding"])
### EXPLORATION ###


# Get unique alternation types for each principal part
unique_alternations_by_pp = data_drop.groupby("principal_part")["C_V_coding"].unique()

# Count the number of unique alternation types for each principal part
alternation_counts_by_pp = unique_alternations_by_pp.apply(len)

print("Number of unique alternation types by principal part:")
print(alternation_counts_by_pp)

# Show which alternation types appear for each principal part
print("\nAlternation types by principal part:")
print(unique_alternations_by_pp)

# Additional analysis: Distribution of alternation types by principal part
alternation_distribution = (
    data_drop.groupby(["principal_part", "C_V_coding"]).size().unstack()
)

alternation_distribution["total"] = alternation_distribution.sum(
    axis=1
)  # Add column of row sums
alternation_distribution.loc["total"] = (
    alternation_distribution.sum()
)  # Add row of column sums

print("\nDistribution of alternation types by principal part:")
print(alternation_distribution)


unique_alternations_by_lemma = data_drop.groupby(["lemma", "principal_part"])[
    "C_V_coding"
].unique()

for lemma in set(data_drop["lemma"]):
    alternations = unique_alternations_by_lemma[lemma]
    if len(alternations) > 1:
        print(f"Lemma: {lemma}")
        print("Alternation types:")
        for pp, alternation in alternations.items():
            print(f"  Principal part {pp}: {alternation}")
        print()


#######
