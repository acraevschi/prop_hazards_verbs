library(rstan)
library(dplyr)
library(tidyr)

# Set Stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Read and prepare data
prepare_stan_data <- function(obs_csv, seq_csv) {
  obs <- read.csv(obs_csv)
  seqs <- read.csv(seq_csv)

  # Map sequences
  seq_ids <- unique(obs$seq_id)
  M <- length(seq_ids)  # number of sequences
  T_len <- integer(M)
  max_T <- 0

  # We'll build matrices: obs_states[M, max_T], time_intervals[M, max_T-1]
  for (m in 1:M) {
    sid <- seq_ids[m]
    Tm <- sum(obs$seq_id == sid)
    T_len[m] <- Tm
    if (Tm > max_T) max_T <- Tm
  }

  obs_states <- matrix(1, nrow=M, ncol=max_T) # observed states at each timepoint
  time_intervals <- matrix(0, nrow=M, ncol=max_T-1) # interval from t to t+1
  form_freq <- matrix(0.0, nrow=M, ncol=max_T)
  lemma_freq <- matrix(0.0, nrow=M, ncol=max_T)
  variety <- integer(M)
  corpus <- integer(M)
  principal_part <- integer(M)
  lemma_id <- integer(M)

  for (m in 1:M) {
    sid <- seq_ids[m]
    sub <- obs[obs$seq_id == sid, ]
    sub <- sub[order(sub$obs_index), ]
    Tm <- nrow(sub)
    obs_states[m, 1:Tm] <- sub$state
    if (Tm > 1) {
      time_intervals[m, 1:(Tm-1)] <- sub$time_interval_to_next[1:(Tm-1)]
    }
    form_freq[m, 1:Tm] <- sub$form_freq
    lemma_freq[m, 1:Tm] <- sub$lemma_freq
    variety[m] <- unique(sub$variety)[1]
    corpus[m] <- unique(sub$corpus)[1]
    principal_part[m] <- unique(sub$principal_part)[1]
    lemma_id[m] <- unique(sub$lemma_id)[1]
  }

  stan_data <- list(
    M = M,
    max_T = max_T,
    T_len = T_len,
    obs_states = obs_states,
    time_intervals = time_intervals,
    form_freq = form_freq,
    lemma_freq = lemma_freq,
    lemma_id = lemma_id,
    principal_part = principal_part,
    variety = variety,
    corpus = corpus,
    N_lemmas = length(unique(lemma_id)),
    N_varieties = max(variety),
    N_corpora = max(corpus),
    alpha_prior = rep(1, 4)
  )

  return(stan_data)
}


prepare_stan_data_aggregated <- function(obs_csv, seq_csv) {
  obs <- read.csv(obs_csv, stringsAsFactors = FALSE)
  seqs <- read.csv(seq_csv, stringsAsFactors = FALSE)

  # sequence ids
  seq_ids <- unique(obs$seq_id)
  M <- length(seq_ids)
  T_len <- integer(M)
  max_T <- 0

  for (m in seq_along(seq_ids)) {
    sid <- seq_ids[m]
    Tm <- sum(obs$seq_id == sid)
    T_len[m] <- Tm
    if (Tm > max_T) max_T <- Tm
  }

  # arrays: counts[M, max_T, 4], totals, time_intervals, covariates
  counts <- array(0L, dim = c(M, max_T, 4))
  totals  <- matrix(0L, nrow = M, ncol = max_T)
  time_intervals <- matrix(0.0, nrow = M, ncol = max_T - 1)
  form_freq <- matrix(0.0, nrow = M, ncol = max_T)
  lemma_freq <- matrix(0.0, nrow = M, ncol = max_T)
  prop_bipartite <- matrix(0.0, nrow = M, ncol = max_T)

  variety <- integer(M)
  corpus <- integer(M)
  principal_part <- integer(M)
  lemma_id <- integer(M)

  for (m in seq_along(seq_ids)) {
    sid <- seq_ids[m]
    sub <- obs[obs$seq_id == sid, ]
    sub <- sub[order(sub$obs_index), ]
    Tm <- nrow(sub)
    T_len[m] <- Tm
    # counts
    counts[m, 1:Tm, 1] <- as.integer(sub$n1)
    counts[m, 1:Tm, 2] <- as.integer(sub$n2)
    counts[m, 1:Tm, 3] <- as.integer(sub$n3)
    counts[m, 1:Tm, 4] <- as.integer(sub$n4)
    totals[m, 1:Tm] <- as.integer(sub$n_total)
    if (Tm > 1) {
      time_intervals[m, 1:(Tm-1)] <- as.numeric(sub$time_interval_to_next[1:(Tm-1)])
    }
    form_freq[m, 1:Tm] <- as.numeric(sub$avg_form_freq)
    lemma_freq[m, 1:Tm] <- as.numeric(sub$avg_lemma_freq)
    prop_bipartite[m, 1:Tm] <- as.numeric(sub$prop_bipartite)
    variety[m] <- if (!all(is.na(sub$variety_code))) unique(sub$variety_code)[1] else 1L
    corpus[m] <- if (!all(is.na(sub$corpus_code))) unique(sub$corpus_code)[1] else 1L
    principal_part[m] <- unique(sub$principal_part)[1]
    lemma_id[m] <- unique(sub$lemma_id)[1]
  }

  stan_data <- list(
    M = M,
    max_T = max_T,
    T_len = T_len,
    counts = counts,    # integer[M, max_T, 4]
    totals = totals,    # integer[M, max_T]
    time_intervals = time_intervals,
    form_freq = log(form_freq),
    lemma_freq = log(lemma_freq),
    prop_bipartite = prop_bipartite,
    lemma_id = lemma_id,
    principal_part = principal_part,
    variety = variety,
    corpus = corpus,
    N_lemmas = length(unique(lemma_id)),
    N_varieties = max(variety, na.rm = TRUE),
    N_corpora = max(corpus, na.rm = TRUE),
    alpha_prior = rep(1.0, 4)
  )

  return(stan_data)
}

# Run Stan model (rest of the function remains the same)
run_analysis <- function(obs_file, seq_file, model_file, output_file = NA, aggregated = FALSE){
  if (is.na(output_file)) {
    file_name <- basename(model_file)
    file_name <- strsplit(file_name, split=".", fixed=T)[[1]][1]
    output_file <- paste0("analysis/results/", file_name, "_fit.rds")
  }

  if (!dir.exists("analysis/results")){
      dir.create("analysis/results")
  }

  if (file.exists(output_file)) {
    # ask user if they want to overwrite
    overwrite <- readline(prompt = paste("Output file", output_file, "already exists. Overwrite? (y/n): "))
    if (tolower(overwrite) != "y") {
      stop("Exiting without overwriting existing file.")
    }
    file.remove(output_file)
  }
    

  # Prepare data
  if (aggregated) {
    stan_data <- prepare_stan_data_aggregated(obs_file, seq_file)
  } else {
    stan_data <- prepare_stan_data(obs_file, seq_file)
  }
  
  # Compile and run model
  model <- stan_model(file = model_file)
  
  # Run sampling
  fit <- sampling(
    model,
    data = stan_data,
    iter = 3000,
    chains = 4,
    warmup = 1000,
    seed = 97,
    refresh = 100
  )
  
  # Save results
  saveRDS(fit, output_file)
  
  return(fit)
}

# Allow running from command line with arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 3) {
  obs_file <- args[1]
  seq_file <- args[2]
  model_file <- args[3]
  output_file <- if (length(args) >= 4) args[4] else NA
  aggregated <- if (length(args) >= 5) as.logical(args[5]) else TRUE

  fit <- run_analysis(
    obs_file = obs_file,
    seq_file = seq_file,
    model_file = model_file,
    output_file = output_file,
    aggregated = aggregated
  )
}
