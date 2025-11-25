import pandas as pd
import numpy as np
from data.extract_modern_freqs import get_lemma_frequencies

# 1. CONFIGURATION
# ---------------------------------------------------------
LEMMA_STANDARDIZATION_MAP = {
    "ziehen": "ziehen",
    "zièhen": "ziehen",
    "zîhen": "ziehen",
    "lieren-": "verlieren",
    "lièsen": "verlieren",
    "verlieren": "verlieren",
    "fliehen": "fliehen",
    "vlièhen": "fliehen",
    "kiesen2": "kiesen",
    "kièsen": "kiesen",
    "kiesen": "kiesen",
    "zeihen": "zeihen",
    "leihen": "leihen",
    "lîhen": "leihen",
    "frieren": "frieren",
    "vrièsen": "frieren",
    "deihen-": "gedeihen",
    "dîhen": "gedeihen",
    "seihen": "seihen",
    "sîhen": "seihen",
    "rîsen": "reisen",
    "rîhen": "rihen",
    "nièsen": "niesen",
}

EXPLICIT_DIALECT_MAP = {
    "Schwaeb": "Upper German",
    "Oschwaeb": "Upper German",
    "oberdeutsch": "Upper German",
    "Els": "Upper German",
    "niederdeutsch": "Low German",
    "OstND": "Low German",
    "mitteldeutsch, niederdeutsch": "Low German",
    "niederdeutsch, hochdeutsch": "Low German",
    "Rip": "Low German",
    "Obs": "Low German",
    "OstF": "Low German",
    "hochdeutsch": "High German",
    "mitteldeutsch": "High German",
    "mitteldeutsch, oberdeutsch": "High German",
    "Hess": "High German",
    "Thuer": "High German",
    "Ofr": "High German",
    "Ostf": "High German",
    "Ohchal": "High German",
    "Eastph": "High German",
    "Mbair": "High German",
}

INPUT_FILE = "data/verner_data_for_analysis.tsv"
BIPARTITE_FILE = "data/lemma_bipartite.csv"
OUTPUT_FILE = "analysis/pauls_principle_dataset.csv"

# 2. LOAD DATA
# ---------------------------------------------------------
print("Loading data...")
df = pd.read_csv(INPUT_FILE, sep="\t")
bipartite_df = pd.read_csv(BIPARTITE_FILE)

# 3. STANDARDIZATION
# ---------------------------------------------------------
print("Standardizing...")
df["lemma_std"] = df["lemma"].apply(
    lambda x: LEMMA_STANDARDIZATION_MAP.get(x.strip(), np.nan)
)
df.dropna(subset=["dialect/place"], inplace=True)
df["dialect"] = df["dialect/place"].apply(
    lambda x: EXPLICIT_DIALECT_MAP.get(x.strip(), np.nan)
)
df = df.dropna(subset=["lemma_std", "dialect"])

# 4. RESTRUCTURING DATA (Wide to Long)
# ---------------------------------------------------------
print("Splitting into element-wise observations (C vs V)...")

observations = []

for idx, row in df.iterrows():
    # Common metadata for this form
    base_data = {
        "lemma_std": row["lemma_std"],
        "lemma": row["lemma"],  # needed for concatenation
        "dialect": row["dialect"],
        "date": row["date"],
        "form": row["form"],
        "inflcat": row["inflcat"],
    }

    # -- Handle Consonant Observation --
    # Only create a row if there is coding present for C
    if pd.notna(row.get("C.coding")) and str(row.get("C.coding")).strip() != "":
        c_data = base_data.copy()
        c_data["element_type"] = "Consonant"
        c_data["has_levelled"] = (
            1 if str(row["C.coding"]).strip().lower() == "innovative" else 0
        )
        observations.append(c_data)

    # -- Handle Vowel Observation --
    # Only create a row if there is coding present for V
    if pd.notna(row.get("V.coding")) and str(row.get("V.coding")).strip() != "":
        v_data = base_data.copy()
        v_data["element_type"] = "Vowel"
        v_data["has_levelled"] = (
            1 if str(row["V.coding"]).strip().lower() == "innovative" else 0
        )
        observations.append(v_data)

# Create new long-format dataframe
long_df = pd.DataFrame(observations)

# 5. FREQUENCY & BIPARTITE MERGE
# ---------------------------------------------------------
print("Merging frequency and bipartite stats...")

# Frequency
lemma_freq_dict = get_lemma_frequencies(list(long_df["lemma_std"].unique()))
# Handle 0 freq
for lemma in lemma_freq_dict.keys():
    if lemma_freq_dict[lemma] == 0:
        lemma_freq_dict[lemma] = 1

long_df["raw_freq"] = long_df["lemma_std"].map(lemma_freq_dict).astype(int)
long_df["log_freq"] = np.log(long_df["raw_freq"])

# Merge Bipartite (Left merge to keep all observations)
long_df = long_df.merge(
    bipartite_df[["lemma", "is_bipartite"]],
    left_on="lemma",
    right_on="lemma",
    how="left",
)

# 6. FINAL CLEANUP
# ---------------------------------------------------------
long_df["date"] = pd.to_numeric(long_df["date"], errors="coerce")
long_df = long_df.dropna(subset=["date", "has_levelled"])

# Columns for R
model_columns = [
    "has_levelled",  # The outcome (0 or 1)
    "element_type",  # The predictor (Consonant or Vowel) - NEW
    "is_bipartite",  # Paradigm feature
    "dialect",
    "date",
    "lemma_std",
    "form",  # Included for inspection
    "inflcat",  # Included for random effects or inspection
    "log_freq",
]

final_df = long_df[model_columns]
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"Exported {len(final_df)} observations to {OUTPUT_FILE}")
