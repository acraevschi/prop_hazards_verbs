# ==============================================================================
# Testing Paul’s Principle: Diachronic Paradigm Leveling in High German Strong Verbs
# ==============================================================================
# Script: analysis/run_brms.R
# Purpose: Fits Bayesian Generalized Additive Mixed Models (GAMMs) via brms / Stan.
#
# Configurable CLI Options (with default fallbacks):
#   --chains <int>         Number of MCMC chains [default: 4]
#   --iter <int>           Total iterations per chain [default: 4000]
#   --warmup <int>         Warmup iterations per chain [default: iter / 2 = 2000]
#   --cores <int>          Number of CPU cores [default: 4]
#   --threads <int>        Within-chain threads [default: 4]
#   --adapt_delta <num>    Target acceptance rate [default: 0.99]
#   --max_treedepth <int>  Maximum NUTS tree depth [default: 10]
#   --backend <str>        Stan backend ("cmdstanr" or "rstan") [default: "cmdstanr"]
#   -h, --help             Show help message and exit
#
# Model Suite Overview (Evaluated in Paper Table 2):
# ------------------------------------------------------------------------------
# 1. Primary Model:
#    - Tensor (k=10): `tensor_fit_marking_type_k10.rds`
#      Full 2D tensor product interaction between time (date) and log lemma frequency.
#
# 2. Sensitivity & Dimension Checks:
#    - Smooth Interaction (k=10): `base_fit_marking_type_k10.rds` (additive smooths, k=10)
#    - Smooth Interaction (k=4): `base_fit_marking_type.rds` (additive smooths, k=4)
#    - Tensor (k=4): `tensor_fit_marking_type_k4.rds` (tensor product with restricted k=4)
#
# 3. Frequency Operationalization Check:
#    - Tensor | Token Freq (k=10): `tensor_fit_marking_type_k10_token.rds`
#      Evaluates token frequency (cell-level) vs. lemma frequency.
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. CLI Argument Parsing
# ------------------------------------------------------------------------------
parse_cli_args <- function(cli_args) {
  params <- list(
    chains = 4,
    iter = 4000,
    warmup = NULL,
    cores = 4,
    threads = 4,
    adapt_delta = 0.99,
    max_treedepth = 10,
    backend = "cmdstanr"
  )
  
  i <- 1
  while (i <= length(cli_args)) {
    arg <- cli_args[i]
    if (arg %in% c("-h", "--help")) {
      cat("Usage: Rscript analysis/run_brms.R [options]\n\n")
      cat("Options:\n")
      cat("  --chains <int>         Number of MCMC chains [default: 4]\n")
      cat("  --iter <int>           Total iterations per chain [default: 4000]\n")
      cat("  --warmup <int>         Warmup iterations per chain [default: iter / 2]\n")
      cat("  --cores <int>          Number of CPU cores [default: 4]\n")
      cat("  --threads <int>        Within-chain threads [default: 4]\n")
      cat("  --adapt_delta <num>    Target acceptance rate [default: 0.99]\n")
      cat("  --max_treedepth <int>  Maximum NUTS tree depth [default: 10]\n")
      cat("  --backend <str>        Stan backend (cmdstanr or rstan) [default: cmdstanr]\n")
      cat("  -h, --help             Show this help message and exit\n\n")
      quit(status = 0)
    } else if (grepl("^--chains=", arg)) {
      params$chains <- as.integer(sub("^--chains=", "", arg))
    } else if (arg == "--chains" && i < length(cli_args)) {
      i <- i + 1; params$chains <- as.integer(cli_args[i])
    } else if (grepl("^--iter=", arg)) {
      params$iter <- as.integer(sub("^--iter=", "", arg))
    } else if (arg == "--iter" && i < length(cli_args)) {
      i <- i + 1; params$iter <- as.integer(cli_args[i])
    } else if (grepl("^--warmup=", arg)) {
      params$warmup <- as.integer(sub("^--warmup=", "", arg))
    } else if (arg == "--warmup" && i < length(cli_args)) {
      i <- i + 1; params$warmup <- as.integer(cli_args[i])
    } else if (grepl("^--cores=", arg)) {
      params$cores <- as.integer(sub("^--cores=", "", arg))
    } else if (arg == "--cores" && i < length(cli_args)) {
      i <- i + 1; params$cores <- as.integer(cli_args[i])
    } else if (grepl("^--threads=", arg)) {
      params$threads <- as.integer(sub("^--threads=", "", arg))
    } else if (arg == "--threads" && i < length(cli_args)) {
      i <- i + 1; params$threads <- as.integer(cli_args[i])
    } else if (grepl("^--adapt_delta=", arg)) {
      params$adapt_delta <- as.numeric(sub("^--adapt_delta=", "", arg))
    } else if (arg == "--adapt_delta" && i < length(cli_args)) {
      i <- i + 1; params$adapt_delta <- as.numeric(cli_args[i])
    } else if (grepl("^--max_treedepth=", arg)) {
      params$max_treedepth <- as.integer(sub("^--max_treedepth=", "", arg))
    } else if (arg == "--max_treedepth" && i < length(cli_args)) {
      i <- i + 1; params$max_treedepth <- as.integer(cli_args[i])
    } else if (grepl("^--backend=", arg)) {
      params$backend <- sub("^--backend=", "", arg)
    } else if (arg == "--backend" && i < length(cli_args)) {
      i <- i + 1; params$backend <- cli_args[i]
    }
    i <- i + 1
  }
  
  if (is.null(params$warmup)) {
    params$warmup <- as.integer(params$iter / 2)
  }
  
  return(params)
}

cfg <- parse_cli_args(commandArgs(trailingOnly = TRUE))

cat("==============================================================================\n")
cat("Bayesian GAMM Estimation: Hermann Paul's Principle in High German Strong Verbs\n")
cat("==============================================================================\n")
cat(sprintf("MCMC Configuration:\n"))
cat(sprintf(" - Chains: %d | Iterations: %d | Warmup: %d\n", cfg$chains, cfg$iter, cfg$warmup))
cat(sprintf(" - Cores: %d  | Threads/Chain: %d | Backend: %s\n", cfg$cores, cfg$threads, cfg$backend))
cat(sprintf(" - adapt_delta: %.3f | max_treedepth: %d\n", cfg$adapt_delta, cfg$max_treedepth))
cat("==============================================================================\n\n")

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(brms)
  library(cmdstanr)
  library(rstan)
})

# ------------------------------------------------------------------------------
# 1. Data Loading & Frequency Aggregation
# ------------------------------------------------------------------------------
cat("Loading coded historical data from data/coded_output.csv...\n")
raw_data <- read.csv("data/coded_output.csv", stringsAsFactors = FALSE)

# Create a lookup for a representative surface lemma (shortest string) per lemma_id
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

# Helper function to compute type frequency (unique lemmas per alternation pattern in each dialect)
calc_type_freq <- function(df, alt_col) {
  df %>%
    filter(!is.na(!!sym(alt_col)), !!sym(alt_col) != "") %>%
    group_by(variety, !!sym(alt_col)) %>%
    summarise(freq = n_distinct(lemma_id), .groups = "drop") %>%
    rename(!!paste0(alt_col, "_freq") := freq)
}

# Attach alternation type frequencies to the dataset
raw_data <- raw_data %>%
  left_join(calc_type_freq(raw_data, "vowel_alternation_pres"), by = c("variety", "vowel_alternation_pres")) %>%
  left_join(calc_type_freq(raw_data, "vowel_alternation_past"), by = c("variety", "vowel_alternation_past")) %>%
  left_join(calc_type_freq(raw_data, "cons_alternation_pres"), by = c("variety", "cons_alternation_pres")) %>%
  left_join(calc_type_freq(raw_data, "cons_alternation_past"), by = c("variety", "cons_alternation_past"))

# ------------------------------------------------------------------------------
# 2. Reshaping & Predictor Construction
# ------------------------------------------------------------------------------
cat("Reshaping dataset into binary leveling observations...\n")
base_model_data <- raw_data %>%
  mutate(
    v_pres = as.numeric(is_leveled_vowel_pres),
    v_past = as.numeric(is_leveled_vowel_past),
    c_pres = as.numeric(is_leveled_cons_pres),
    c_past = as.numeric(is_leveled_cons_past),
    # Combine leveling status across present-to-past and internal past singular/plural shifts
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

# Construct 3-level joint marking_type predictor to avoid structural collinearity
model_data <- base_model_data %>%
  mutate(
    log_alt_pres_freq = if_else(has_alt_pres == "yes", log(target_alt_pres_freq), 0),
    log_alt_past_freq = if_else(has_alt_past == "yes", log(target_alt_past_freq), 0),
    marking_type = case_when(
      element_type == "vowel" & is_bipartite %in% c(0, "0") ~ "vowel_unipartite",
      element_type == "vowel" & is_bipartite %in% c(1, "1") ~ "vowel_bipartite",
      element_type == "consonant" & is_bipartite %in% c(1, "1") ~ "consonant_bipartite"
    ),
    # Ensure factor structures
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
cat(sprintf("Prepared %d modeling observations across %d unique lemmas.\n", nrow(model_data), n_distinct(model_data$lemma_std)))

# Save prepared analysis dataset
write.csv(model_data, "analysis/data_for_analysis.csv", row.names = FALSE)

# ------------------------------------------------------------------------------
# 3. Priors & Shared MCMC Sampler Settings
# ------------------------------------------------------------------------------

# Weakly informative priors regularizing log-odds and variance parameters
priors <- c(
  prior(normal(0, 1.5), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
  prior(exponential(2), class = "sd"),
  prior(exponential(2), class = "sds")
)

# Standardized MCMC sampler settings
mcmc_control <- list(adapt_delta = cfg$adapt_delta, max_treedepth = cfg$max_treedepth)
threads <- threading(cfg$threads)

# ------------------------------------------------------------------------------
# 4. Model Estimation
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Model 1: Smooth Interaction (k=4) [Appendix Baseline]
# ------------------------------------------------------------------------------
cat("\n[1/5] Estimating Model 1: Smooth Interaction GAMM (k=4)...\n")
formula_base_k4 <- bf(
  has_levelled ~
    s(date, k = 4) +
    marking_type +
    s(date, by = marking_type, k = 4) +
    log_freq + s(date, by = log_freq, k = 4) +
    has_alt_pres + log_alt_pres_freq +
    has_alt_past + log_alt_past_freq +
    std_infl + s(date, by = std_infl, k = 4) +
    std_infl * marking_type +
    (1 | variety) + s(date, by = variety) +
    (1 | lemma_std) +
    (1 | id),
  family = bernoulli()
)

fit_base_k4 <- brm(
  formula = formula_base_k4,
  data = model_data,
  prior = priors,
  chains = cfg$chains, iter = cfg$iter, warmup = cfg$warmup,
  cores = cfg$cores, threads = threads, backend = cfg$backend,
  control = mcmc_control,
  file = "fits/base_fit_marking_type"
)
add_criterion(fit_base_k4, "loo")

# ------------------------------------------------------------------------------
# Model 2: Smooth Interaction (k=10)
# ------------------------------------------------------------------------------
cat("\n[2/5] Estimating Model 2: Smooth Interaction GAMM (k=10)...\n")
formula_base_k10 <- bf(
  has_levelled ~
    s(date, k = 10) +
    marking_type +
    s(date, by = marking_type, k = 10) +
    log_freq + s(date, by = log_freq, k = 10) +
    has_alt_pres + log_alt_pres_freq +
    has_alt_past + log_alt_past_freq +
    std_infl + s(date, by = std_infl, k = 10) +
    std_infl * marking_type +
    (1 | variety) + s(date, by = variety) +
    (1 | lemma_std) +
    (1 | id),
  family = bernoulli()
)

fit_base_k10 <- brm(
  formula = formula_base_k10,
  data = model_data,
  prior = priors,
  chains = cfg$chains, iter = cfg$iter, warmup = cfg$warmup,
  cores = cfg$cores, threads = threads, backend = cfg$backend,
  control = mcmc_control,
  file = "fits/base_fit_marking_type_k10"
)
add_criterion(fit_base_k10, "loo")

# ------------------------------------------------------------------------------
# Model 3: Primary Tensor Product GAMM (k=10) [Primary Model in Paper]
# ------------------------------------------------------------------------------
cat("\n[3/5] Estimating Model 3: Primary Tensor Product GAMM (k=10)...\n")
formula_tensor_k10 <- bf(
  has_levelled ~
    s(date, k = 10) +
    marking_type +
    s(date, by = marking_type, k = 10) +
    log_freq + t2(date, log_freq, k = 10) +
    has_alt_pres + log_alt_pres_freq +
    has_alt_past + log_alt_past_freq +
    std_infl + s(date, by = std_infl, k = 10) +
    std_infl * marking_type +
    (1 | variety) + s(date, by = variety) +
    (1 | lemma_std) +
    (1 | id),
  family = bernoulli()
)

fit_tensor_k10 <- brm(
  formula = formula_tensor_k10,
  data = model_data,
  prior = priors,
  chains = cfg$chains, iter = cfg$iter, warmup = cfg$warmup,
  cores = cfg$cores, threads = threads, backend = cfg$backend,
  control = mcmc_control,
  file = "fits/tensor_fit_marking_type_k10"
)
add_criterion(fit_tensor_k10, "loo")

# ------------------------------------------------------------------------------
# Model 4: Tensor Product GAMM (k=4) [Sensitivity Check on Basis Dimension]
# ------------------------------------------------------------------------------
cat("\n[4/5] Estimating Model 4: Tensor Product GAMM (k=4)...\n")
formula_tensor_k4 <- bf(
  has_levelled ~
    s(date, k = 4) +
    marking_type +
    s(date, by = marking_type, k = 4) +
    log_freq + t2(date, log_freq, k = 4) +
    has_alt_pres + log_alt_pres_freq +
    has_alt_past + log_alt_past_freq +
    std_infl + s(date, by = std_infl, k = 4) +
    std_infl * marking_type +
    (1 | variety) + s(date, by = variety) +
    (1 | lemma_std) +
    (1 | id),
  family = bernoulli()
)

fit_tensor_k4 <- brm(
  formula = formula_tensor_k4,
  data = model_data,
  prior = priors,
  chains = cfg$chains, iter = cfg$iter, warmup = cfg$warmup,
  cores = cfg$cores, threads = threads, backend = cfg$backend,
  control = mcmc_control,
  file = "fits/tensor_fit_marking_type_k4"
)
add_criterion(fit_tensor_k4, "loo")

# ------------------------------------------------------------------------------
# Model 5: Tensor Product with Token Frequency (k=10) [Sensitivity Check on Frequency]
# ------------------------------------------------------------------------------
cat("\n[5/5] Estimating Model 5: Tensor Product with Token Frequency (k=10)...\n")
formula_tensor_token_k10 <- bf(
  has_levelled ~
    s(date, k = 10) +
    marking_type +
    s(date, by = marking_type, k = 10) +
    log_token_freq + t2(date, log_token_freq, k = 10) +
    has_alt_pres + log_alt_pres_freq +
    has_alt_past + log_alt_past_freq +
    std_infl + s(date, by = std_infl, k = 10) +
    std_infl * marking_type +
    (1 | variety) + s(date, by = variety) +
    (1 | lemma_std) +
    (1 | id),
  family = bernoulli()
)

fit_tensor_token_k10 <- brm(
  formula = formula_tensor_token_k10,
  data = model_data,
  prior = priors,
  chains = cfg$chains, iter = cfg$iter, warmup = cfg$warmup,
  cores = cfg$cores, threads = threads, backend = cfg$backend,
  control = mcmc_control,
  file = "fits/tensor_fit_marking_type_k10_token"
)
add_criterion(fit_tensor_token_k10, "loo")

cat("\nAll 5 models fitted/verified and cached in fits/ successfully!\n")
