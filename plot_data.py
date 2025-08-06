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

data_drop = data.dropna(subset=["date", "dialect", "principal_part", "C_V_coding"])

data_drop.columns

### PLOTTING ###

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

# Filter for principal parts 1-4 only
filtered_data = data_drop[data_drop["principal_part"].isin([1, 2, 3, 4])]

# Get unique lemmas
unique_lemmas = filtered_data["lemma"].unique()

# Set up colors for C_V_coding
color_map = {
    "conservative": "blue",
    "innovative": "red",
    "V_innovative": "orange",
    "C_innovative": "purple",
}


# Function to plot data for a specific lemma
def plot_lemma_data(lemma, ax):
    lemma_data = filtered_data[filtered_data["lemma"] == lemma]

    # Plot horizontal lines for each principal part
    for pp in [1, 2, 3, 4]:
        ax.axhline(y=pp, color="gray", linestyle="--", alpha=0.3)

    # Plot points
    for _, row in lemma_data.iterrows():
        jitter = np.random.uniform(-0.2, 0.2)
        ax.scatter(
            row["date"],
            row["principal_part"] + jitter,
            color=color_map.get(row["C_V_coding"], "gray"),
            s=50,
            alpha=0.7,
        )

    # Set labels and title
    ax.set_title(f"Lemma: {lemma}")
    ax.set_xlabel("")
    ax.set_ylabel("Principal Part")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0.5, 4.5)


# Create a grid of plots based on number of lemmas
num_lemmas = len(unique_lemmas)
cols = min(4, num_lemmas)  # Maximum 3 columns
rows = int(np.ceil(num_lemmas / cols))

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten() if num_lemmas > 1 else [axes]

# Plot each lemma
for i, lemma in enumerate(unique_lemmas):
    if i < len(axes):
        plot_lemma_data(lemma, axes[i])

# Hide unused subplots
for j in range(num_lemmas, len(axes)):
    axes[j].set_visible(False)

# Add legend
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=color,
        label=label,
        markersize=10,
    )
    for label, color in color_map.items()
]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.05),
    ncol=len(color_map),
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.05)  # Make room for the legend
plt.savefig("data/lemma_principal_parts_timeline.png", dpi=300)
# plt.show()
