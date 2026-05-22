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
  select(-lemma_rep) %>%
  group_by(lemma_id, std_infl) %>%
  mutate(token_freq_avg = mean(form_freq_per_1000, na.rm = TRUE)) %>%
  ungroup()


# 2. Calculate Alternation Frequencies (Type Frequency by Variety)
# ------------------------------------
# A helper function to count unique lemmas for a given alternation pattern per variety
calc_type_freq <- function(df, alt_col) {
  df %>%
    filter(!is.na(!!sym(alt_col)), !!sym(alt_col) != "") %>%
    group_by(variety, !!sym(alt_col)) %>%
    summarise(freq = n_distinct(lemma_id), .groups = "drop") %>%
    rename(!!paste0(alt_col, "_freq") := freq)
}

# Attach the frequencies to the main dataset
raw_data <- raw_data %>%
  left_join(calc_type_freq(raw_data, "vowel_alternation_pres"), by = c("variety", "vowel_alternation_pres")) %>%
  left_join(calc_type_freq(raw_data, "vowel_alternation_past"), by = c("variety", "vowel_alternation_past")) %>%
  left_join(calc_type_freq(raw_data, "cons_alternation_pres"), by = c("variety", "cons_alternation_pres")) %>%
  left_join(calc_type_freq(raw_data, "cons_alternation_past"), by = c("variety", "cons_alternation_past"))


# 3. Preprocessing Pipeline
# ------------------------------------
base_model_data <- raw_data %>%
  mutate(
    v_pres = as.numeric(is_leveled_vowel_pres),
    v_past = as.numeric(is_leveled_vowel_past),
    c_pres = as.numeric(is_leveled_cons_pres),
    c_past = as.numeric(is_leveled_cons_past),
    vowel_leveled_any = case_when(
      v_pres == 1 | v_past == 1 ~ 1,
      v_pres == 0 | v_past == 0 ~ 0,
      TRUE ~ NA_real_
    ),
    cons_leveled_any = case_when(
      c_pres == 1 | c_past == 1 ~ 1,
      c_pres == 0 | c_past == 0 ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  filter(!is.na(is_bipartite)) %>%
  filter(!is.na(vowel_leveled_any) | !is.na(cons_leveled_any)) %>%
  pivot_longer(
    cols = c(vowel_leveled_any, cons_leveled_any),
    names_to = "element_type_raw",
    values_to = "has_levelled"
  ) %>%
  filter(!is.na(has_levelled)) %>%
  mutate(
    element_type = if_else(element_type_raw == "vowel_leveled_any", "vowel", "consonant"),
    target_alt_pres_freq = if_else(element_type == "vowel", vowel_alternation_pres_freq, cons_alternation_pres_freq),
    target_alt_past_freq = if_else(element_type == "vowel", vowel_alternation_past_freq, cons_alternation_past_freq),
    has_alt_pres = if_else(!is.na(target_alt_pres_freq) & target_alt_pres_freq > 0, "yes", "no"),
    has_alt_past = if_else(!is.na(target_alt_past_freq) & target_alt_past_freq > 0, "yes", "no"),
    log_freq = log(lemma_freq_per_1000 + 0.0001),
    log_token_freq = log(token_freq_avg + 0.0001)
  )

# Step 3b: Calculate the GLOBAL means for centering (only for existing alternations)
# We log it first, then take the mean, to get the geometric mean of the frequencies.
global_mean_pres <- mean(log(base_model_data$target_alt_pres_freq[base_model_data$has_alt_pres == "yes"]))
global_mean_past <- mean(log(base_model_data$target_alt_past_freq[base_model_data$has_alt_past == "yes"]))

# Step 3c: Apply the centering and finalize factors
model_data <- base_model_data %>%
  mutate(
    # NEW: Apply the global mean centering. If "no", exactly 0.
    # log_alt_pres_freq = if_else(has_alt_pres == "yes", log(target_alt_pres_freq) - global_mean_pres, 0),
    # log_alt_past_freq = if_else(has_alt_past == "yes", log(target_alt_past_freq) - global_mean_past, 0),
    log_alt_pres_freq = if_else(has_alt_pres == "yes", log(target_alt_pres_freq), 0),
    log_alt_past_freq = if_else(has_alt_past == "yes", log(target_alt_past_freq), 0),
    marking_type = case_when(
      element_type == "vowel" & is_bipartite %in% c(0, "0") ~ "vowel_unipartite",
      element_type == "vowel" & is_bipartite %in% c(1, "1") ~ "vowel_bipartite",
      element_type == "consonant" & is_bipartite %in% c(1, "1") ~ "consonant_bipartite"
    ),

    # Ensure factors
    has_alt_pres = as.factor(has_alt_pres),
    has_alt_past = as.factor(has_alt_past),
    marking_type = as.factor(marking_type),
    element_type = as.factor(element_type),
    is_bipartite = as.factor(is_bipartite),
    variety = as.factor(variety),
    corpus = as.factor(corpus),
    lemma_std = as.factor(lemma_id),
    id = as.factor(id),
    std_infl = as.factor(std_infl)
  ) %>%
  select(
    lemma, lemma_std, date, log_freq, log_token_freq,
    has_alt_pres, log_alt_pres_freq,
    has_alt_past, log_alt_past_freq,
    marking_type, is_bipartite, element_type, has_levelled,
    id, variety, std_infl, corpus
  )

model_data <- unique(model_data)
# write.csv(model_data, "analysis/data_for_analysis.csv", row.names = FALSE)

# 4. Model definition and running
# ------------------------------------

### PRIORS ###
priors <- c(
  prior(normal(0, 1.5), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
  prior(exponential(2), class = "sd"),
  prior(exponential(2), class = "sds")
)

### FORMULA ###
base_formula <- bf(
  has_levelled ~
    s(date, k = 4) +
    marking_type +
    s(date, by = marking_type, k = 4) +
    log_freq + s(date, by = log_freq, k = 4) +

    # NEW: Include BOTH the indicator and the continuous frequency
    has_alt_pres + log_alt_pres_freq +
    has_alt_past + log_alt_past_freq +

    std_infl + s(date, by = std_infl, k = 4) +
    std_infl * marking_type +
    (1 | variety) + s(date, by = variety) +
    (1 | lemma_std) +
    (1 | id),
  family = bernoulli()
)

### FITTING ###
fit <- brm(
  formula = base_formula,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = 4,
  threads = threading(4),
  backend = "cmdstanr",
  control = list(adapt_delta = 0.99, max_treedepth = 10),
  file = "fits/base_fit_marking_type"
)

add_criterion(fit, "loo")


### FORMULA ###
base_formula_k10 <- bf(
  has_levelled ~
    s(date, k = 10) +
    marking_type +
    s(date, by = marking_type, k = 10) +
    log_freq + s(date, by = log_freq, k = 10) +

    # NEW: Include BOTH the indicator and the continuous frequency
    has_alt_pres + log_alt_pres_freq +
    has_alt_past + log_alt_past_freq +

    std_infl + s(date, by = std_infl, k = 10) +
    std_infl * marking_type +
    (1 | variety) + s(date, by = variety) +
    (1 | lemma_std) +
    (1 | id),
  family = bernoulli()
)

### FITTING ###
fit <- brm(
  formula = base_formula_k10,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = 4,
  threads = threading(4),
  backend = "cmdstanr",
  control = list(adapt_delta = 0.99, max_treedepth = 10),
  file = "fits/base_fit_marking_type_k10"
)

add_criterion(fit, "loo")

### FORMULA ###
tensor_formula_k10 <- bf(
  has_levelled ~
    s(date, k = 10) +
    marking_type +
    s(date, by = marking_type, k = 10) +
    log_freq + t2(date, log_freq, k = 10) +

    # NEW: Include BOTH the indicator and the continuous frequency
    has_alt_pres + log_alt_pres_freq +
    has_alt_past + log_alt_past_freq +

    std_infl + s(date, by = std_infl, k = 10) +
    std_infl * marking_type +
    (1 | variety) + s(date, by = variety) +
    (1 | lemma_std) +
    (1 | id),
  family = bernoulli()
)

### FITTING ###
fit <- brm(
  formula = tensor_formula_k10,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = 4,
  threads = threading(4),
  backend = "cmdstanr",
  control = list(adapt_delta = 0.99, max_treedepth = 10),
  file = "fits/tensor_fit_marking_type_k10"
)

add_criterion(fit, "loo")


### FORMULA ###
tensor_formula_k4 <- bf(
  has_levelled ~
    s(date, k = 4) +
    marking_type +
    s(date, by = marking_type, k = 4) +
    log_freq + t2(date, log_freq, k = 4) +

    # NEW: Include BOTH the indicator and the continuous frequency
    has_alt_pres + log_alt_pres_freq +
    has_alt_past + log_alt_past_freq +

    std_infl + s(date, by = std_infl, k = 4) +
    std_infl * marking_type +
    (1 | variety) + s(date, by = variety) +
    (1 | lemma_std) +
    (1 | id),
  family = bernoulli()
)

### FITTING ###
fit <- brm(
  formula = tensor_formula_k4,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = 4,
  threads = threading(4),
  backend = "cmdstanr",
  control = list(adapt_delta = 0.99, max_treedepth = 10),
  file = "fits/tensor_fit_marking_type_k4"
)

add_criterion(fit, "loo")


### FORMULA ###
tensor_formula_k10_token <- bf(
  has_levelled ~
    s(date, k = 10) +
    marking_type +
    s(date, by = marking_type, k = 10) +
    log_token_freq + t2(date, log_token_freq, k = 10) +

    # NEW: Include BOTH the indicator and the continuous frequency
    has_alt_pres + log_alt_pres_freq +
    has_alt_past + log_alt_past_freq +

    std_infl + s(date, by = std_infl, k = 10) +
    std_infl * marking_type +
    (1 | variety) + s(date, by = variety) +
    (1 | lemma_std) +
    (1 | id),
  family = bernoulli()
)

### FITTING ###
fit <- brm(
  formula = tensor_formula_k10_token,
  data = model_data,
  prior = priors,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = 4,
  threads = threading(4),
  backend = "cmdstanr",
  control = list(adapt_delta = 0.99, max_treedepth = 10),
  file = "fits/tensor_fit_marking_type_k10_token"
)

add_criterion(fit, "loo")
