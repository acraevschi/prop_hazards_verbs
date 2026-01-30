library(dplyr)
library(tidyr)
library(brms)
library(cmdstanr)
library(rstan)


# 1. Load Data
# ------------------------------------
raw_data <- read.csv("data/coded_output.csv", stringsAsFactors = FALSE)

# Create a lookup for a representative lemma (shortest string) per lemma_id
lemma_lookup <- raw_data %>%
  filter(!is.na(lemma), lemma != "") %>%
  group_by(lemma_id) %>%
  slice_min(nchar(lemma), n = 1, with_ties = FALSE) %>%
  select(lemma_id, lemma_rep = lemma) %>%
  ungroup()

raw_data <- raw_data %>%
  left_join(lemma_lookup, by = "lemma_id") %>%
  mutate(lemma = lemma_rep) %>%
  select(-lemma_rep)

model_data <- raw_data %>%
  # A. Filter: Keep only rows where is_bipartite is coded (not NA)
  filter(!is.na(is_bipartite)) %>%
  
  # B. Filter: Drop rows where BOTH leveling indicators are missing
  #    (Keep row if at least one of them has a value)
  filter(!is.na(is_leveled_vowel_pres) | !is.na(is_leveled_cons_pres))


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
  select(lemma, lemma_std, date, id, variety, std_infl, 
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
    # interaction between dialect and time
    s(date, by = variety) + 
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


priors <- c(
  # A. Intercept
  # Normal(0, 1.5) on the log-odds scale.
  # This covers a probability range of roughly 0.05 to 0.95, which is realistic 
  # for leveling (it's rarely 0% or 100% likely across the whole board).
  prior(normal(0, 1.5), class = "Intercept"),
  
  # B. Fixed Effects (Betas)
  # Normal(0, 1). This assumes that the effect of any single predictor (like bipartite)
  # is unlikely to shift the log-odds by more than +/- 2 (odds ratios of 0.13 to 7.4).
  # This constrains the model from finding "exploded" coefficients due to separation.
  prior(normal(0, 1), class = "b"),
  
  # C. Random Effects SDs (Group-level variations)
  # Exponential(2). This penalizes very large standard deviations.
  # It assumes most groups (dialects, lemmas) are clustered relatively close to the average,
  # but allows for exceptions if the data strongly supports it.
  prior(exponential(2), class = "sd"),
  
  # D. Smooths (Splines)
  # Exponential(2). Controls the "wiggliness" of the time trajectories.
  # Prevents the curve from overfitting every minor fluctuation in the centuries.
  prior(exponential(2), class = "sds"),
  
  # E. Correlations (for lemma_std random slopes)
  # LKJ(2). The LKJ prior controls the correlation matrix.
  # A value of 1 is uniform (any correlation is equally likely).
  # A value of 2 weakly favors 0 correlation, making the model skeptical of 
  # perfect correlations (-1 or 1) unless the data screams for it.
  prior(lkj(2), class = "cor")
)

# 4. Model Fitting
# ------------------------------------
# Note: This is a complex model. Ensure you have sufficient data points per group 
# to support the random slopes, otherwise convergence warnings may occur.
comprehensive_fit <- brm(
  formula = comprehensive_formula,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4500,           # Increased iterations for complex random effects
  warmup = 3250,
  cores = 4,
  threads = threading(2),
  backend = "cmdstanr",     # or "cmdstanr" if installed for speed
  control = list(adapt_delta = 0.99, max_treedepth=12), # Stricter controls for convergence
  file = "analysis/models/comprehensive_brms_fit"
)


#########################################################################################################

comprehensive_formula_no_slopes <- bf(
  has_levelled ~ 
    # --- Fixed Effects & Interactions ---
    is_bipartite + 
    element_type +
    # Smooths for Time Interactions
    # How the leveling probability changes over time, specific to Vowel vs Consonant
    s(date, by = element_type) + 
    # How the leveling probability changes over time, specific to Bipartite vs Non-Bipartite
    s(date, by = is_bipartite) +
    # interaction between dialect and time
    s(date, by = variety) + 
    # Interaction between Time and Frequency (Tensor product smooth)
    # Allows the effect of frequency to vary over time (e.g., freq effects might get stronger later)
    t2(date, log_freq) +
    # --- Random Effects ---
    # 1. Document ID as random intercept
    (1 | id) +
    # 2. Variety/Dialect as random intercept
    (1 | variety) +
    # 3. Inflectional Context as random intercept
    (1 | std_infl),
  family = bernoulli()
)

priors <- c(
  # A. Intercept
  # Normal(0, 1.5) on the log-odds scale.
  # This covers a probability range of roughly 0.05 to 0.95, which is realistic 
  # for leveling (it's rarely 0% or 100% likely across the whole board).
  prior(normal(0, 1.5), class = "Intercept"),
  
  # B. Fixed Effects (Betas)
  # Normal(0, 1). This assumes that the effect of any single predictor (like bipartite)
  # is unlikely to shift the log-odds by more than +/- 2 (odds ratios of 0.13 to 7.4).
  # This constrains the model from finding "exploded" coefficients due to separation.
  prior(normal(0, 1), class = "b"),
  
  # C. Random Effects SDs (Group-level variations)
  # Exponential(2). This penalizes very large standard deviations.
  # It assumes most groups (dialects, lemmas) are clustered relatively close to the average,
  # but allows for exceptions if the data strongly supports it.
  prior(exponential(2), class = "sd"),
  
  # D. Smooths (Splines)
  # Exponential(2). Controls the "wiggliness" of the time trajectories.
  # Prevents the curve from overfitting every minor fluctuation in the centuries.
  prior(exponential(2), class = "sds")
)

comprehensive_fit_no_slopes <- brm(
  formula = comprehensive_formula_no_slopes,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4500,           # Increased iterations for complex random effects
  warmup = 3250,
  cores = 4,
  threads = threading(2),
  backend = "cmdstanr",     # or "cmdstanr" if installed for speed
  control = list(adapt_delta = 0.99, max_treedepth=12), # Stricter controls for convergence
  file = "analysis/models/comprehensive_brms_fit_no_slopes"
)



#################################################################################################

comprehensive_formula_k3 <- bf(
  has_levelled ~ 
    # --- Fixed Effects & Interactions ---
    is_bipartite + 
    element_type +
    # Smooths for Time Interactions
    # How the leveling probability changes over time, specific to Vowel vs Consonant
    s(date, by = element_type, k=3) + 
    # How the leveling probability changes over time, specific to Bipartite vs Non-Bipartite
    s(date, by = is_bipartite, k=3) +
    # interaction between dialect and time
    s(date, by = variety, k=3) + 
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


priors <- c(
  # A. Intercept
  # Normal(0, 1.5) on the log-odds scale.
  # This covers a probability range of roughly 0.05 to 0.95, which is realistic 
  # for leveling (it's rarely 0% or 100% likely across the whole board).
  prior(normal(0, 1.5), class = "Intercept"),
  
  # B. Fixed Effects (Betas)
  # Normal(0, 1). This assumes that the effect of any single predictor (like bipartite)
  # is unlikely to shift the log-odds by more than +/- 2 (odds ratios of 0.13 to 7.4).
  # This constrains the model from finding "exploded" coefficients due to separation.
  prior(normal(0, 1), class = "b"),
  
  # C. Random Effects SDs (Group-level variations)
  # Exponential(2). This penalizes very large standard deviations.
  # It assumes most groups (dialects, lemmas) are clustered relatively close to the average,
  # but allows for exceptions if the data strongly supports it.
  prior(exponential(2), class = "sd"),
  
  # D. Smooths (Splines)
  # Exponential(2). Controls the "wiggliness" of the time trajectories.
  # Prevents the curve from overfitting every minor fluctuation in the centuries.
  prior(exponential(2), class = "sds"),
  
  # E. Correlations (for lemma_std random slopes)
  # LKJ(2). The LKJ prior controls the correlation matrix.
  # A value of 1 is uniform (any correlation is equally likely).
  # A value of 2 weakly favors 0 correlation, making the model skeptical of 
  # perfect correlations (-1 or 1) unless the data screams for it.
  prior(lkj(2), class = "cor")
)

# 4. Model Fitting
# ------------------------------------
# Note: This is a complex model. Ensure you have sufficient data points per group 
# to support the random slopes, otherwise convergence warnings may occur.
comprehensive_fit_k3 <- brm(
  formula = comprehensive_formula_k3,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4000,           # Increased iterations for complex random effects
  warmup = 2750,
  cores = 4,
  threads = threading(2),
  backend = "cmdstanr",     # or "cmdstanr" if installed for speed
  control = list(adapt_delta = 0.99, max_treedepth=12), # Stricter controls for convergence
  file = "analysis/models/comprehensive_brms_fit_k3"
)

#########################################################################################################


linear_freq_formula_k3 <- bf(
  has_levelled ~ 
    # --- Fixed Effects & Interactions ---
    is_bipartite + 
    element_type +
    # Smooths for Time Interactions
    # How the leveling probability changes over time, specific to Vowel vs Consonant
    s(date, by = element_type, k=3) + 
    # How the leveling probability changes over time, specific to Bipartite vs Non-Bipartite
    s(date, by = is_bipartite, k=3) +
    # interaction between dialect and time
    s(date, by = variety, k=3) + 
    # The "Average" linear effect of frequency and how that effect changes over time
    log_freq + s(date, by = log_freq) +
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


priors <- c(
  # A. Intercept
  # Normal(0, 1.5) on the log-odds scale.
  # This covers a probability range of roughly 0.05 to 0.95, which is realistic 
  # for leveling (it's rarely 0% or 100% likely across the whole board).
  prior(normal(0, 1.5), class = "Intercept"),
  
  # B. Fixed Effects (Betas)
  # Normal(0, 1). This assumes that the effect of any single predictor (like bipartite)
  # is unlikely to shift the log-odds by more than +/- 2 (odds ratios of 0.13 to 7.4).
  # This constrains the model from finding "exploded" coefficients due to separation.
  prior(normal(0, 1), class = "b"),
  
  # C. Random Effects SDs (Group-level variations)
  # Exponential(2). This penalizes very large standard deviations.
  # It assumes most groups (dialects, lemmas) are clustered relatively close to the average,
  # but allows for exceptions if the data strongly supports it.
  prior(exponential(2), class = "sd"),
  
  # D. Smooths (Splines)
  # Exponential(2). Controls the "wiggliness" of the time trajectories.
  # Prevents the curve from overfitting every minor fluctuation in the centuries.
  prior(exponential(2), class = "sds"),
  
  # E. Correlations (for lemma_std random slopes)
  # LKJ(2). The LKJ prior controls the correlation matrix.
  # A value of 1 is uniform (any correlation is equally likely).
  # A value of 2 weakly favors 0 correlation, making the model skeptical of 
  # perfect correlations (-1 or 1) unless the data screams for it.
  prior(lkj(2), class = "cor")
)

# 4. Model Fitting
# ------------------------------------
# Note: This is a complex model. Ensure you have sufficient data points per group 
# to support the random slopes, otherwise convergence warnings may occur.
linear_freq_fit_k3 <- brm(
  formula = linear_freq_formula_k3,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4000,           # Increased iterations for complex random effects
  warmup = 2750,
  cores = 4,
  threads = threading(2),
  backend = "cmdstanr",     # or "cmdstanr" if installed for speed
  control = list(adapt_delta = 0.99, max_treedepth=10), # Stricter controls for convergence
  file = "analysis/models/linear_freq_brms_fit_k3"
)
