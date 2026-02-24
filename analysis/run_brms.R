### Try model stacking in `loo` (loo_model_weights)

### Try to fit the model on MHG-only and on ENHG-only and then compare the results

### Marginal effect of freq on levelling. Take into account how the freq contributes to all the parts where it participates

### Do the plot of tensor product of freq and date to see if it shows any pattern

### Start writing part of the paper: the background, the rationale. This will make clear justification of what we're making
### It would be good to lay everything out down to choosing tensor product vs smooth; see the previous effects of freq
### in other papers, cite those as expectations for our paper
### 



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
# write.csv(model_data, "analysis/data_for_analysis.csv", row.names = FALSE)


# 3. Model definition and running

### PRIORS ###
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
)

### FORMULA ###

base_formula <- bf(
  has_levelled ~ 
    s(date, k = 4) +
    is_bipartite +
    s(date, by = is_bipartite, k = 4),
    element_type + s(date, by = element_type, k = 4) +
    element_type * is_bipartite +
    log_freq + s(date, by = log_freq, k = 4) + 
    std_infl + s(date, by = std_infl, k = 4) + 
    std_infl * element_type + 
    (1|variety) + s(date, by = variety) + 
    (1|lemma_std) +
    (1|id),    
  family = bernoulli()
)

### FITTING ###

fit <- brm(
  formula = base_formula,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4000,           # Increased iterations for complex random effects
  warmup = 2000,
  cores = 4,
  threads = threading(2),
  backend = "cmdstanr",     # allows for threading
  control = list(adapt_delta = 0.99, max_treedepth=10), # Slightly stricter controls for convergence
  file = "analysis/models/base_fit"
)



### FORMULA 2 ###

base_formula_k10 <- bf(
  has_levelled ~ 
    s(date, k = 10) +
    is_bipartite +
    s(date, by = is_bipartite, k = 10),
    element_type + s(date, by = element_type, k = 10) +
    element_type * is_bipartite +
    log_freq + s(date, by = log_freq, k = 10) + 
    std_infl + s(date, by = std_infl, k = 10) + 
    std_infl * element_type + 
    (1|variety) + s(date, by = variety) + 
    (1|lemma_std) +
    (1|id),    
  family = bernoulli()
)

### FITTING ###

fit <- brm(
  formula = base_formula_k10,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4000,           # Increased iterations for complex random effects
  warmup = 2000,
  cores = 4,
  threads = threading(2),
  backend = "cmdstanr",     # allows for threading
  control = list(adapt_delta = 0.99, max_treedepth=10), # Slightly stricter controls for convergence
  file = "analysis/models/base_fit_k10"
)



### FORMULA ###

base_formula_tensor_product <- bf(
  has_levelled ~ 
    s(date, k = 4) +
    is_bipartite +
    s(date, by = is_bipartite, k = 4),
    element_type + s(date, by = element_type, k = 4) +
    element_type * is_bipartite +
    log_freq + t2(date, log_freq, k = c(4, 4)) + 
    std_infl + s(date, by = std_infl, k = 4) + 
    std_infl * element_type + 
    (1|variety) + s(date, by = variety) + 
    (1|lemma_std) +
    (1|id),    
  family = bernoulli()
)

### FITTING ###

fit <- brm(
  formula = base_formula_tensor_product,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4500,           # Increased iterations for complex random effects
  warmup = 2500,
  cores = 4,
  threads = threading(2),
  backend = "cmdstanr",     # allows for threading
  control = list(adapt_delta = 0.99, max_treedepth=10), # Slightly stricter controls for convergence
  file = "analysis/models/base_fit_tensor_product"
)