import pandas as pd
import numpy as np
from data.extract_modern_freqs import get_lemma_frequencies

# 1. CONFIGURATION & DICTIONARIES
# ---------------------------------------------------------

# Lemma standardization mappings
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
    # Upper German / Alemannic
    "Schwaeb": "Upper German",
    "Oschwaeb": "Upper German",
    "oberdeutsch": "Upper German",
    "Els": "Upper German",
    # Low German
    "niederdeutsch": "Low German",
    "OstND": "Low German",
    "mitteldeutsch, niederdeutsch": "Low German",
    "niederdeutsch, hochdeutsch": "Low German",
    "Rip": "Low German",
    "Obs": "Low German",
    "OstF": "Low German",
    # High German
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
print("Standardizing lemmas and dialects...")

# Apply Lemma Map
# We use 'get' to return the original if not found, or you can map to NaN to filter
df["lemma_std"] = df["lemma"].apply(
    lambda x: LEMMA_STANDARDIZATION_MAP.get(x.strip(), np.nan)
)

# Apply Dialect Map
df.dropna(subset=["dialect/place"], inplace=True)
df["dialect"] = df["dialect/place"].apply(
    lambda x: EXPLICIT_DIALECT_MAP.get(x.strip(), np.nan)
)

# Drop rows where lemma or dialect could not be standardized
initial_count = len(df)
df = df.dropna(subset=["lemma_std", "dialect"])
print(f"Filtered {initial_count - len(df)} rows due to unmapped lemmas or dialects.")

# 4. CODING 'HAS_LEVELLED' (THE RESPONSE VARIABLE)
# ---------------------------------------------------------
print("Coding leveling status...")


# Logic: If EITHER Consonant (C) or Vowel (V) is 'innovative', the form has suffered leveling.
def determine_leveling(row):
    # Check for 'innovative' string in coding columns (case insensitive)
    c_innovative = str(row.get("C.coding", "")).strip().lower() == "innovative"
    v_innovative = str(row.get("V.coding", "")).strip().lower() == "innovative"

    if c_innovative or v_innovative:
        return 1
    return 0


df["has_levelled"] = df.apply(determine_leveling, axis=1)

# 5. FREQUENCY CALCULATION (log_freq)
# ---------------------------------------------------------
print("Calculating log frequency...")

lemma_freq_dict = get_lemma_frequencies(list(df["lemma_std"].unique()))

# to account for 0 modern freq
for lemma in lemma_freq_dict.keys():
    if lemma_freq_dict[lemma] == 0:
        lemma_freq_dict[lemma] = 1

df["raw_freq"] = df["lemma_std"].map(lemma_freq_dict).astype(int)

# Apply Log transformation: log(freq)
df["log_freq"] = np.log(df["raw_freq"])

# 6. MERGE BIPARTITE PREDICTORS
# ---------------------------------------------------------
print("Merging bipartite data...")

# Merge left to keep all observations in the main data
df = df.merge(bipartite_df[["lemma", "is_bipartite"]], on="lemma", how="left")

# 7. FINAL CLEANUP AND EXPORT
# ---------------------------------------------------------
print("Finalizing dataset...")

# Ensure Date is numeric
df["date"] = pd.to_numeric(df["date"], errors="coerce")
df = df.dropna(subset=["date"])  # Drop rows with invalid dates

# Select only columns needed for the brms model
model_columns = [
    "has_levelled",
    "is_bipartite",
    "dialect",
    "date",
    "lemma_std",
    "log_freq",
]

final_df = df[model_columns]

# Save
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"Success! Dataset saved to {OUTPUT_FILE} with {len(final_df)} rows.")
print(final_df.head())
