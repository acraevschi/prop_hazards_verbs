library(dplyr)
library(tidyr)
library(brms)
library(cmdstanr)


# 1. Load Data
# ------------------------------------
raw_data <- read.csv("data/coded_output.csv", stringsAsFactors = FALSE)

# 2. Preprocessing Pipeline
# ------------------------------------
model_data <- raw_data %>%
  # A. Filter: Keep only rows where is_bipartite is coded (not NA)
  filter(!is.na(is_bipartite)) %>%
  
  # B. Filter: Drop rows where BOTH leveling indicators are missing
  #    (Keep row if at least one of them has a value)
  filter(!is.na(is_leveled_vowel_pres) | !is.na(is_leveled_cons_pres)) %>%
  
  # C. Transformation: Pivot to Long Format
  #    We want separate rows for Vowel leveling events and Consonant leveling events
  pivot_longer(
    cols = c(is_leveled_vowel_pres, is_leveled_cons_pres),
    names_to = "element_type_raw",
    values_to = "has_levelled"
  ) %>%
  
  # D. Post-Pivot Cleaning
  #    Remove rows created by the pivot that are NAs (e.g. if a verb only had vowel info)
  filter(!is.na(has_levelled)) %>%
  mutate(
    # Clean up element type names
    element_type = if_else(element_type_raw == "is_leveled_vowel_pres", "vowel", "consonant"),
    
    # Ensure factors
    element_type = as.factor(element_type),
    is_bipartite = as.factor(is_bipartite),
    variety = as.factor(variety),
    lemma_std = as.factor(lemma_id), # Assuming 'lemma' column is the standard ID
    id = as.factor(id),           # Document ID
    std_infl = as.factor(std_infl),
    
    # Log transform frequency (adding small constant to avoid log(0) if needed)
    log_freq = log(lemma_freq_per_1000 + 0.0001)
  ) %>%
  
  # Select final columns for cleanliness
  select(lemma_std, date, id, variety, std_infl, 
         log_freq, is_bipartite, element_type, has_levelled)


model_data <- unique(model_data)


# 3. The "Mega-Model" Definition
# ------------------------------------
# 1. Interaction date * element_type -> s(date, by = element_type)
# 2. Interaction date * is_bipartite -> s(date, by = is_bipartite)
# 3. Interaction date * freq         -> t2(date, log_freq)
# 4. Random Effects:
#    - Variety (intercept)
#    - Document ID (intercept)
#    - Std_infl (intercept)
#    - Lemma: Random Intercept + Random Slopes for element_type and is_bipartite

comprehensive_formula <- bf(
  has_levelled ~ 
    # --- Fixed Effects & Interactions ---
    is_bipartite + 
    element_type +
    # Smooths for Time Interactions
    # How the leveling probability changes over time, specific to Vowel vs Consonant
    s(date, by = element_type) + 
    # How the leveling probability changes over time, specific to Bipartite vs Non-Bipartite
    s(date, by = is_bipartite) +
    # Interaction between Time and Frequency (Tensor product smooth)
    # Allows the effect of frequency to vary over time (e.g., freq effects might get stronger later)
    t2(date, log_freq) +
    # --- Random Effects ---
    # 1. Document ID as random intercept
    (1 | id) +
    # 2. Variety/Dialect as random intercept
    (1 | variety) +
    # 3. Inflectional Context as random intercept
    (1 | std_infl) +
    # 4. Lemma Random Structure
    #    Intercept + Slopes for Element Type and Bipartiteness.
    #    This accounts for specific verbs being more prone to vowel/consonant leveling
    #    or reacting differently to the bipartite constraint.
    (1 + element_type + is_bipartite | lemma_std),
    
  family = bernoulli()
)

# 4. Model Fitting
# ------------------------------------
# Note: This is a complex model. Ensure you have sufficient data points per group 
# to support the random slopes, otherwise convergence warnings may occur.
comprehensive_fit <- brm(
  formula = comprehensive_formula,
  data = model_data,
  chains = 3,
  iter = 4000,           # Increased iterations for complex random effects
  warmup = 2500,
  cores = 3,
  threads = threading(2),
  backend = "cmdstanr",     # or "cmdstanr" if installed for speed
  control = list(adapt_delta = 0.99, max_treedepth=12), # Stricter controls for convergence
  file = "analysis/models/comprehensive_brms_fit"
)


