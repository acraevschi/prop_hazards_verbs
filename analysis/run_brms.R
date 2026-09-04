# ==============================================================================
# Testing Paul’s Principle: Diachronic Paradigm Leveling in High German Strong Verbs
# ==============================================================================
# Script: analysis/run_brms.R
# Purpose: Fits Bayesian Generalized Additive Mixed Models (GAMMs) via brms / Stan.
#          Primary analysis uses a Vowel-Only Model with unipartite vowels as the
#          reference baseline (intercept), testing whether bipartite vowel marking
#          resists analogical leveling (Paul's Principle).
#
# Configurable CLI Options (with default fallbacks):
#   --chains <int>         Number of MCMC chains [default: 4]
#   --iter <int>           Total iterations per chain [default: 4000]
#   --warmup <int>         Warmup iterations per chain [default: iter / 2 = 2000]
#   --cores <int>          Number of chains to run in parallel [default: 4]
#   --threads <int>        Within-chain threads [default: 2]
#   --seed <int>           Random seed [default: 97]
#   --adapt_delta <num>    Target acceptance rate [default: 0.99]
#   --max_treedepth <int>  Maximum NUTS tree depth [default: 10]
#   --backend <str>        Stan backend ("cmdstanr" or "rstan") [default: "cmdstanr"]
#   --overwrite, -o        Refit models that already exist in fits/ [default: FALSE]
#   --test                 Quick test run with small iterations/chains [default: FALSE]
#   --dry-run              Validate stancode & data without MCMC sampling [default: FALSE]
#   -h, --help             Show help message and exit
#
# Reproducibility: the seed fixes the results only because within-chain threading
# runs in static mode (see `threading(..., static = TRUE)` below). Dynamic
# scheduling changes the order of the floating-point reduction between runs.
#
# Model Suite Overview (Evaluated in Paper Table 2):
# ------------------------------------------------------------------------------
# 1. Primary Model:
#    - Tensor | Token Freq (k=10): `tensor_fit_marking_type_k10_token.rds`
#      Full 2D tensor product interaction between time (date) and log token frequency.
#      Vowel-only observations with vowel_unipartite reference baseline (highest LOO ELPD).
#
# 2. Dimension & Frequency Sensitivity Checks:
#    - Tensor | Token Freq (k=4): `tensor_fit_marking_type_k4_token.rds` (token frequency, k=4)
#    - Tensor | Lemma Freq (k=10): `tensor_fit_marking_type_k10.rds` (lemma frequency, k=10)
#    - Tensor | Lemma Freq (k=4): `tensor_fit_marking_type_k4.rds` (lemma frequency, k=4)
#    - Smooth Interaction (k=10): `base_fit_marking_type_k10.rds` (additive smooths, k=10)
#    - Smooth Interaction (k=4): `base_fit_marking_type.rds` (additive smooths, k=4 baseline)
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. CLI Argument Parsing
# ------------------------------------------------------------------------------
# Option table: name -> default value and value type.
CLI_DEFAULTS <- list(
  chains = 4L,
  iter = 4000L,
  warmup = NULL,
  cores = 4L,
  threads = 2L,
  seed = 97L,
  adapt_delta = 0.99,
  max_treedepth = 10L,
  backend = "cmdstanr",
  overwrite = FALSE,
  test = FALSE,
  dry_run = FALSE,
  model = "all"
)

CLI_TYPES <- c(
  chains = "int", iter = "int", warmup = "int", cores = "int",
  threads = "int", seed = "int", adapt_delta = "num",
  max_treedepth = "int", backend = "str", overwrite = "flag",
  test = "flag", dry_run = "flag", model = "str"
)

print_cli_help <- function() {
  cat("Usage: Rscript analysis/run_brms.R [options]\n\n")
  cat("Options:\n")
  cat("  --chains <int>         Number of MCMC chains [default: 4]\n")
  cat("  --iter <int>           Total iterations per chain [default: 4000]\n")
  cat("  --warmup <int>         Warmup iterations per chain [default: iter / 2]\n")
  cat("  --cores <int>          Number of chains to run in parallel [default: 4]\n")
  cat("  --threads <int>        Within-chain threads [default: 2]\n")
  cat("  --seed <int>           Random seed [default: 97]\n")
  cat("  --adapt_delta <num>    Target acceptance rate [default: 0.99]\n")
  cat("  --max_treedepth <int>  Maximum NUTS tree depth [default: 10]\n")
  cat("  --backend <str>        Stan backend (cmdstanr or rstan) [default: cmdstanr]\n")
  cat("  --overwrite, -o        Refit models that already exist in fits/ [default: FALSE]\n")
  cat("  --test                 Quick test run with small iterations/chains [default: FALSE]\n")
  cat("  --dry-run              Validate formulas & Stan code without sampling [default: FALSE]\n")
  cat("  --model <str>          Which model to fit (1-6, or all) [default: all]\n")
  cat("  -h, --help             Show this help message and exit\n\n")
  cat("Booleans accept true/false, yes/no, and 1/0. You can also write\n")
  cat("--no-overwrite, --no-test, etc.\n\n")
}

# Convert one raw string to the type that the option needs. Stop on bad input.
coerce_cli_value <- function(name, raw, type) {
  if (type == "str") {
    return(raw)
  }
  if (type == "flag") {
    truthy <- c("true", "t", "yes", "y", "1")
    falsy <- c("false", "f", "no", "n", "0")
    low <- tolower(raw)
    if (low %in% truthy) return(TRUE)
    if (low %in% falsy) return(FALSE)
    stop(sprintf("Option --%s needs true or false, but got '%s'.", name, raw),
         call. = FALSE)
  }
  value <- suppressWarnings(
    if (type == "int") as.integer(raw) else as.numeric(raw)
  )
  if (is.na(value)) {
    stop(sprintf("Option --%s needs a number, but got '%s'.", name, raw),
         call. = FALSE)
  }
  value
}

parse_cli_args <- function(cli_args) {
  params <- CLI_DEFAULTS
  flag_words <- c("true", "t", "yes", "y", "1", "false", "f", "no", "n", "0")

  i <- 1
  while (i <= length(cli_args)) {
    arg <- cli_args[i]

    if (arg %in% c("-h", "--help")) {
      print_cli_help()
      quit(status = 0)
    }
    if (arg == "-o") {
      arg <- "--overwrite"
    }
    if (arg == "--dryrun") {
      arg <- "--dry-run"
    }
    if (!grepl("^--[A-Za-z]", arg)) {
      stop(sprintf("Unexpected argument '%s'. Run with --help for the options.", arg),
           call. = FALSE)
    }

    # Split the --name=value form from the --name value form.
    inline <- grepl("=", arg, fixed = TRUE)
    name <- if (inline) sub("^--([^=]+)=.*$", "\\1", arg) else sub("^--", "", arg)
    name <- gsub("-", "_", name)
    raw <- if (inline) sub("^--[^=]+=", "", arg) else NA_character_

    # Handle the --no-<flag> form.
    negated <- FALSE
    if (!name %in% names(CLI_TYPES) && grepl("^no_", name)) {
      stripped <- sub("^no_", "", name)
      if (stripped %in% names(CLI_TYPES) && CLI_TYPES[[stripped]] == "flag") {
        name <- stripped
        negated <- TRUE
      }
    }

    if (!name %in% names(CLI_TYPES)) {
      stop(sprintf("Unknown option '--%s'. Run with --help for the options.", name),
           call. = FALSE)
    }
    type <- CLI_TYPES[[name]]

    if (negated) {
      if (inline) {
        stop(sprintf("Option '--no-%s' does not take a value.", name), call. = FALSE)
      }
      params[[name]] <- FALSE
      i <- i + 1
      next
    }

    if (!inline) {
      if (type == "flag") {
        # A bare flag means TRUE, but --flag true and --flag false also work.
        nxt <- if (i < length(cli_args)) cli_args[i + 1] else NA_character_
        if (!is.na(nxt) && tolower(nxt) %in% flag_words) {
          i <- i + 1
          raw <- nxt
        } else {
          raw <- "true"
        }
      } else {
        if (i >= length(cli_args)) {
          stop(sprintf("Option --%s needs a value.", name), call. = FALSE)
        }
        i <- i + 1
        raw <- cli_args[i]
      }
    }

    params[[name]] <- coerce_cli_value(name, raw, type)
    i <- i + 1
  }

  if (params$test) {
    params$chains <- min(params$chains, 2L)
    params$cores <- min(params$cores, 2L)
    params$iter <- 50L
    params$warmup <- 25L
  } else if (is.null(params$warmup)) {
    params$warmup <- as.integer(params$iter / 2)
  }

  # Reject configurations that Stan cannot run.
  if (params$chains < 1) stop("--chains must be 1 or more.", call. = FALSE)
  if (params$cores < 1) stop("--cores must be 1 or more.", call. = FALSE)
  if (params$threads < 1) stop("--threads must be 1 or more.", call. = FALSE)
  if (params$iter < 2) stop("--iter must be 2 or more.", call. = FALSE)
  if (params$warmup < 1 || params$warmup >= params$iter) {
    stop("--warmup must be 1 or more and less than --iter.", call. = FALSE)
  }
  if (params$adapt_delta <= 0 || params$adapt_delta >= 1) {
    stop("--adapt_delta must be between 0 and 1.", call. = FALSE)
  }
  if (params$max_treedepth < 1) stop("--max_treedepth must be 1 or more.", call. = FALSE)
  if (!params$backend %in% c("cmdstanr", "rstan")) {
    stop(sprintf("--backend must be cmdstanr or rstan, but got '%s'.", params$backend),
         call. = FALSE)
  }

  params
}

cfg <- parse_cli_args(commandArgs(trailingOnly = TRUE))

cat("==============================================================================\n")
cat("Bayesian GAMM Estimation: Hermann Paul's Principle in High German Strong Verbs\n")
cat("  (Option A: Vowel-Only Model with Unipartite Reference Baseline)\n")
cat("==============================================================================\n")
cat(sprintf("MCMC Configuration:\n"))
cat(sprintf(" - Chains: %d | Iterations: %d | Warmup: %d | Seed: %d\n", cfg$chains, cfg$iter, cfg$warmup, cfg$seed))
cat(sprintf(" - Parallel chains: %d | Threads/Chain: %d | Total CPUs: %d | Backend: %s\n",
            min(cfg$chains, cfg$cores), cfg$threads,
            min(cfg$chains, cfg$cores) * cfg$threads, cfg$backend))
cat(sprintf(" - adapt_delta: %.3f | max_treedepth: %d\n", cfg$adapt_delta, cfg$max_treedepth))
cat(sprintf(" - Overwrite existing fits: %s | Test mode: %s | Dry-run: %s\n",
            if (cfg$overwrite) "TRUE (--overwrite)" else "FALSE (skip existing)",
            if (cfg$test) "TRUE" else "FALSE",
            if (cfg$dry_run) "TRUE" else "FALSE"))
cat("==============================================================================\n\n")

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(brms)
  library(cmdstanr)
  library(rstan)
})

# Configure cmdstanr path and environment if available
if (cfg$backend == "cmdstanr") {
  user_cmdstan <- file.path(Sys.getenv("HOME"), ".cmdstan/cmdstan-2.38.0")
  if (dir.exists(user_cmdstan)) {
    cmdstanr::set_cmdstan_path(user_cmdstan)
  }
  tbb_path <- file.path(cmdstanr::cmdstan_path(), "stan/lib/stan_math/lib/tbb")
  if (dir.exists(tbb_path)) {
    Sys.setenv(DYLD_LIBRARY_PATH = paste0(tbb_path, ":", Sys.getenv("DYLD_LIBRARY_PATH")))
  }
}

# ------------------------------------------------------------------------------
# 1. Data Loading & Frequency Aggregation
# ------------------------------------------------------------------------------
cat("Loading coded historical data from data/coded_output.csv...\n")
raw_data <- read.csv("data/coded_output.csv", stringsAsFactors = FALSE)

# Create a lookup for a representative surface lemma (shortest string) per lemma_id
lemma_lookup <- raw_data %>%
  filter(!is.na(lemma), lemma != "") %>%
  group_by(lemma_id) %>%
  # Sort on the string as well as its length. Without the second key the tie
  # between two lemmas of equal length breaks on row order, and the label
  # changes between runs.
  arrange(nchar(lemma), lemma, .by_group = TRUE) %>%
  slice(1) %>%
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
    # calc_type_freq counts distinct lemmas per alternation pattern, so the
    # frequency is 1 or more whenever the join found a match. A `> 0` test would
    # never be false. These two variables record whether corpus_approach_coding.py
    # wrote an alternation pair for the row, which is "no" when the paradigm has
    # no alternation in that contrast, and also "no" when the anchor is missing.
    has_alt_pres = if_else(!is.na(target_alt_pres_freq), "yes", "no"),
    has_alt_past = if_else(!is.na(target_alt_past_freq), "yes", "no"),
    log_freq = log(lemma_freq_per_1000 + 0.0001),
    log_token_freq = log(token_freq_avg + 0.0001)
  )

# Construct joint marking_type predictor and filter to vowel-only modeling dataset
model_data <- base_model_data %>%
  mutate(
    log_alt_pres_freq = if_else(has_alt_pres == "yes", log(target_alt_pres_freq), 0),
    log_alt_past_freq = if_else(has_alt_past == "yes", log(target_alt_past_freq), 0),
    marking_type = case_when(
      element_type == "vowel" & is_bipartite %in% c(0, "0") ~ "vowel_unipartite",
      element_type == "vowel" & is_bipartite %in% c(1, "1") ~ "vowel_bipartite",
      element_type == "consonant" & is_bipartite %in% c(1, "1") ~ "consonant_bipartite"
    )
  ) %>%
  # Filter out consonant rows for the primary vowel-only model
  filter(marking_type %in% c("vowel_unipartite", "vowel_bipartite")) %>%
  mutate(
    # Explicitly set vowel_unipartite as the reference level (baseline intercept)
    marking_type = factor(marking_type, levels = c("vowel_unipartite", "vowel_bipartite")),
    # Set "yes" as reference level for presence of alternations (typical strong verb baseline)
    has_alt_pres = factor(has_alt_pres, levels = c("yes", "no")),
    has_alt_past = factor(has_alt_past, levels = c("yes", "no")),
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
cat(sprintf("Prepared %d vowel-only modeling observations across %d unique lemmas.\n", nrow(model_data), n_distinct(model_data$lemma_std)))

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

# static = TRUE fixes the grainsize and the partition of the reduce_sum call.
# Without it the threads sum the log-likelihood in a different order on each run
# and --seed does not reproduce the fit.
threads <- threading(cfg$threads, static = TRUE)

# Ensure fits/ directory exists
dir.create("fits", showWarnings = FALSE, recursive = TRUE)

# Fit a model, or load it from fits/ when it is already there.
# --overwrite is the only way to refit an existing model.
#
# LOO-CV is the PSIS approximation from loo(). Read the Pareto-k diagnostics in
# analyze_models.Rmd before you trust the model comparison.
fit_and_cache_model <- function(formula, data, priors, cfg, threads, mcmc_control,
                                file_base, model_title) {
  cat(sprintf("\n%s...\n", model_title))
  rds_path <- paste0(file_base, ".rds")

  if (cfg$dry_run) {
    cat(sprintf("[DRY-RUN] Validating Stan code and data structures for %s...\n", file_base))
    sc <- make_stancode(formula = formula, data = data, prior = priors)
    cat(sprintf("[DRY-RUN] Stan code generated successfully (%d characters).\n", nchar(sc)))
    return(invisible(NULL))
  }

  if (file.exists(rds_path) && !cfg$overwrite) {
    cat(sprintf("File '%s' already exists, skipping. Run with --overwrite to refit it.\n", rds_path))
    fit <- readRDS(rds_path)
    if (!"loo" %in% names(fit$criteria)) {
      cat(sprintf("Adding LOO-CV to %s...\n", rds_path))
      fit <- add_criterion(fit, "loo", file = file_base)
    }
    return(fit)
  }

  if (file.exists(rds_path)) {
    cat(sprintf("File '%s' exists, refitting as requested (--overwrite)...\n", rds_path))
  }

  # No `file` argument here. brms would load the cached fit instead of running
  # the sampler, which would cancel --overwrite.
  fit <- brm(
    formula = formula,
    data = data,
    prior = priors,
    chains = cfg$chains, iter = cfg$iter, warmup = cfg$warmup,
    cores = cfg$cores, threads = threads, backend = cfg$backend,
    seed = cfg$seed,
    control = mcmc_control
  )

  # Save before the LOO step, so that a completed fit is never lost.
  saveRDS(fit, rds_path)
  
  if (!cfg$test || brms::ndraws(fit) >= 100) {
    tryCatch({
      fit <- add_criterion(fit, "loo", file = file_base)
    }, error = function(e) {
      cat(sprintf("Warning: Could not compute LOO-CV for %s: %s\n", file_base, e$message))
    })
  }
  
  fit
}

# ------------------------------------------------------------------------------
# 4. Model Estimation
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Model 1: Smooth Interaction (k=4) [Appendix Baseline]
# ------------------------------------------------------------------------------
if (cfg$model %in% c("all", "1", "base_k4")) {
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
      (1 | variety) + s(date, by = variety, k = 4) +
      (1 | lemma_std) +
      (1 | id),
    family = bernoulli()
  )

  fit_base_k4 <- fit_and_cache_model(
    formula = formula_base_k4,
    data = model_data,
    priors = priors,
    cfg = cfg,
    threads = threads,
    mcmc_control = mcmc_control,
    file_base = "fits/base_fit_marking_type",
    model_title = "[1/5] Estimating Model 1: Smooth Interaction GAMM (k=4) [Appendix Baseline]"
  )
}

# ------------------------------------------------------------------------------
# Model 2: Smooth Interaction (k=10)
# ------------------------------------------------------------------------------
if (cfg$model %in% c("all", "2", "base_k10")) {
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
      (1 | variety) + s(date, by = variety, k = 10) +
      (1 | lemma_std) +
      (1 | id),
    family = bernoulli()
  )

  fit_base_k10 <- fit_and_cache_model(
    formula = formula_base_k10,
    data = model_data,
    priors = priors,
    cfg = cfg,
    threads = threads,
    mcmc_control = mcmc_control,
    file_base = "fits/base_fit_marking_type_k10",
    model_title = "[2/5] Estimating Model 2: Smooth Interaction GAMM (k=10)"
  )
}

# ------------------------------------------------------------------------------
# Model 3: Primary Tensor Product GAMM (k=10) [Primary Model in Paper]
# ------------------------------------------------------------------------------
if (cfg$model %in% c("all", "3", "tensor_k10")) {
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
      (1 | variety) + s(date, by = variety, k = 10) +
      (1 | lemma_std) +
      (1 | id),
    family = bernoulli()
  )

  fit_tensor_k10 <- fit_and_cache_model(
    formula = formula_tensor_k10,
    data = model_data,
    priors = priors,
    cfg = cfg,
    threads = threads,
    mcmc_control = mcmc_control,
    file_base = "fits/tensor_fit_marking_type_k10",
    model_title = "[3/5] Estimating Model 3: Primary Tensor Product GAMM (k=10) [Primary Model in Paper]"
  )
}

# ------------------------------------------------------------------------------
# Model 4: Tensor Product GAMM (k=4) [Sensitivity Check on Basis Dimension]
# ------------------------------------------------------------------------------
if (cfg$model %in% c("all", "4", "tensor_k4")) {
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
      (1 | variety) + s(date, by = variety, k = 4) +
      (1 | lemma_std) +
      (1 | id),
    family = bernoulli()
  )

  fit_tensor_k4 <- fit_and_cache_model(
    formula = formula_tensor_k4,
    data = model_data,
    priors = priors,
    cfg = cfg,
    threads = threads,
    mcmc_control = mcmc_control,
    file_base = "fits/tensor_fit_marking_type_k4",
    model_title = "[4/5] Estimating Model 4: Tensor Product GAMM (k=4) [Sensitivity Check on Basis Dimension]"
  )
}

# ------------------------------------------------------------------------------
# Model 5: Tensor Product with Token Frequency (k=10) [Sensitivity Check on Frequency]
# ------------------------------------------------------------------------------
if (cfg$model %in% c("all", "5", "tensor_token", "token", "tensor_token_k10", "token_k10")) {
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
      (1 | variety) + s(date, by = variety, k = 10) +
      (1 | lemma_std) +
      (1 | id),
    family = bernoulli()
  )

  fit_tensor_token_k10 <- fit_and_cache_model(
    formula = formula_tensor_token_k10,
    data = model_data,
    priors = priors,
    cfg = cfg,
    threads = threads,
    mcmc_control = mcmc_control,
    file_base = "fits/tensor_fit_marking_type_k10_token",
    model_title = "[5/6] Estimating Model 5: Tensor Product with Token Frequency (k=10) [Sensitivity Check on Frequency]"
  )
}

# ------------------------------------------------------------------------------
# Model 6: Tensor Product with Token Frequency (k=4) [Sensitivity Check on Basis Dimension + Frequency]
# ------------------------------------------------------------------------------
if (cfg$model %in% c("all", "6", "tensor_token_k4", "token_k4", "token4")) {
  formula_tensor_token_k4 <- bf(
    has_levelled ~
      s(date, k = 4) +
      marking_type +
      s(date, by = marking_type, k = 4) +
      log_token_freq + t2(date, log_token_freq, k = 4) +
      has_alt_pres + log_alt_pres_freq +
      has_alt_past + log_alt_past_freq +
      std_infl + s(date, by = std_infl, k = 4) +
      std_infl * marking_type +
      (1 | variety) + s(date, by = variety, k = 4) +
      (1 | lemma_std) +
      (1 | id),
    family = bernoulli()
  )

  fit_tensor_token_k4 <- fit_and_cache_model(
    formula = formula_tensor_token_k4,
    data = model_data,
    priors = priors,
    cfg = cfg,
    threads = threads,
    mcmc_control = mcmc_control,
    file_base = "fits/tensor_fit_marking_type_k4_token",
    model_title = "[6/6] Estimating Model 6: Tensor Product with Token Frequency (k=4) [Sensitivity Check on Basis Dimension + Frequency]"
  )
}

cat("\nRequested model estimation complete!\n")
