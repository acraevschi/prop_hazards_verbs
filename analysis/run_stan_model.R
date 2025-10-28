library(rstan)
library(dplyr)
library(tidyr)

# Set Stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

prepare_stan_data_aggregated <- function(obs_csv, seq_csv, lemma_freq_column = "avg_lemma_freq", n_sample = NA) {
  obs <- read.csv(obs_csv, stringsAsFactors = FALSE)
  seqs <- read.csv(seq_csv, stringsAsFactors = FALSE)

  # Optionally select top-n most frequent lemmas
  if (!is.na(n_sample) && n_sample > 0L) {
    freq_tbl <- sort(table(obs$lemma_id), decreasing = TRUE)
    top_ids <- as.integer(names(freq_tbl)[seq_len(min(n_sample, length(freq_tbl)))])
    obs <- obs[obs$lemma_id %in% top_ids, , drop = FALSE]
    seqs <- seqs[seqs$lemma_id %in% top_ids, , drop = FALSE]
    if (nrow(obs) == 0) stop("No observations left after filtering by top-n lemma_id")
    message("Using top ", length(unique(obs$lemma_id)), " lemma_id(s) by frequency")
  }

  # remap lemma_id to consecutive integers 1..N_lemmas
  lemma_levels <- sort(unique(obs$lemma_id))
  lemma_map <- setNames(seq_along(lemma_levels), as.character(lemma_levels))
  obs$lemma_id <- as.integer(lemma_map[as.character(obs$lemma_id)])
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

    if (!lemma_freq_column %in% names(sub)) {
      stop(paste0("Column '", lemma_freq_column, "' not found in data"))
    }
    lemma_freq[m, 1:Tm] <- as.numeric(sub[[lemma_freq_column]])

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
    form_freq = (form_freq - mean(form_freq))/sd(form_freq),
    lemma_freq = (lemma_freq - mean(lemma_freq))/sd(lemma_freq),
    prop_bipartite = prop_bipartite,
    lemma_id = lemma_id,
    principal_part = principal_part,
    variety = variety,
    corpus = corpus,
    N_lemmas = length(lemma_levels),
    N_varieties = max(variety, na.rm = TRUE),
    N_corpora = max(corpus, na.rm = TRUE),
    alpha_prior = rep(1.0, 4)
  )

  return(stan_data)
}

prepare_stan_data_transitions <- function(obs_csv, seq_csv, lemma_freq_column = "avg_lemma_freq", n_sample = NA) {
  obs <- read.csv(obs_csv, stringsAsFactors = FALSE)
  seqs <- read.csv(seq_csv, stringsAsFactors = FALSE)

  # Optionally select top-n most frequent lemmas
  if (!is.na(n_sample) && n_sample > 0L) {
    freq_tbl <- sort(table(obs$lemma_id), decreasing = TRUE)
    top_ids <- as.integer(names(freq_tbl)[seq_len(min(n_sample, length(freq_tbl)))])
    obs <- obs[obs$lemma_id %in% top_ids, , drop = FALSE]
    seqs <- seqs[seqs$lemma_id %in% top_ids, , drop = FALSE]
    if (nrow(obs) == 0) stop("No observations left after filtering by top-n lemma_id")
    message("Using top ", length(unique(obs$lemma_id)), " lemma_id(s) by frequency")
  }

  # remap lemma_id to consecutive integers 1..N_lemmas
  lemma_levels <- sort(unique(obs$lemma_id))
  lemma_map <- setNames(seq_along(lemma_levels), as.character(lemma_levels))
  obs$lemma_id <- as.integer(lemma_map[as.character(obs$lemma_id)])

  # Ensure obs sorted by seq_id and obs_index
  obs <- obs[order(obs$seq_id, obs$obs_index), ]

  # For each sequence, create one row per interval t -> t+1
  intervals <- list()
  row_idx <- 1L

  # Collect global lemma_freq values for standardization
  all_lemma_freq_vals <- obs[[lemma_freq_column]]

  for (sid in unique(obs$seq_id)) {
    sub <- obs[obs$seq_id == sid, ]
    sub <- sub[order(sub$obs_index), ]
    Tm <- nrow(sub)
    if (Tm < 2) next  # no interval

    for (t in 1:(Tm-1)) {
      # row for interval from t to t+1
      row <- list()
      # counts at time t (aggregated states)
      row$counts_t <- as.integer(c(sub$n1[t], sub$n2[t], sub$n3[t], sub$n4[t]))
      # counts at time t+1
      row$counts_tp1 <- as.integer(c(sub$n1[t+1], sub$n2[t+1], sub$n3[t+1], sub$n4[t+1]))
      # totals (optional)
      row$total_t <- as.integer(sub$n_total[t])
      row$total_tp1 <- as.integer(sub$n_total[t+1])
      # time interval dt (to next)
      row$dt <- as.numeric(sub$time_interval_to_next[t])
      # covariates from time t (use time t covariates for transition)
      if (!lemma_freq_column %in% names(sub)) stop(paste0("Column '", lemma_freq_column, "' not found"))
      row$lemma_freq <- as.numeric(sub[[lemma_freq_column]][t])
      row$prop_bipartite <- as.numeric(sub$prop_bipartite[t])
      row$variety <- if (!all(is.na(sub$variety_code))) unique(sub$variety_code)[1] else 1L
      row$corpus <- if (!all(is.na(sub$corpus_code))) unique(sub$corpus_code)[1] else 1L
      row$principal_part <- as.integer(sub$principal_part[t]) # principal part for this sequence
      row$lemma_id <- as.integer(sub$lemma_id[t])
      # store sequence id and time index if needed
      row$seq_id <- sid
      row$t_index <- sub$obs_index[t]

      intervals[[row_idx]] <- row
      row_idx <- row_idx + 1L
    }
  }

  # bind to data.frame-like structures
  N_intervals <- length(intervals)
  if (N_intervals == 0) stop("No intervals found in data (need at least one sequence with T >= 2)")

  counts_t <- matrix(0L, nrow = N_intervals, ncol = 4)
  counts_tp1 <- matrix(0L, nrow = N_intervals, ncol = 4)
  dt <- numeric(N_intervals)
  lemma_freq <- numeric(N_intervals)
  prop_bipartite <- numeric(N_intervals)
  variety <- integer(N_intervals)
  corpus <- integer(N_intervals)
  principal_part <- integer(N_intervals)
  lemma_id <- integer(N_intervals)
  seq_id <- integer(N_intervals)
  t_index <- integer(N_intervals)

  for (i in seq_len(N_intervals)) {
    counts_t[i, ] <- intervals[[i]]$counts_t
    counts_tp1[i, ] <- intervals[[i]]$counts_tp1
    dt[i] <- intervals[[i]]$dt
    lemma_freq[i] <- intervals[[i]]$lemma_freq
    prop_bipartite[i] <- intervals[[i]]$prop_bipartite
    variety[i] <- intervals[[i]]$variety
    corpus[i] <- intervals[[i]]$corpus
    principal_part[i] <- intervals[[i]]$principal_part
    lemma_id[i] <- intervals[[i]]$lemma_id
    seq_id[i] <- intervals[[i]]$seq_id
    t_index[i] <- intervals[[i]]$t_index
  }

  # standardize lemma_freq using global mean/sd (like your old code)
  lemma_freq_mean <- mean(all_lemma_freq_vals, na.rm = TRUE)
  lemma_freq_sd <- sd(all_lemma_freq_vals, na.rm = TRUE)
  lemma_freq_std <- (lemma_freq - lemma_freq_mean) / lemma_freq_sd

  stan_data <- list(
    N_intervals = N_intervals,
    counts_t = counts_t,        # int[N_intervals, 4]
    counts_tp1 = counts_tp1,    # int[N_intervals, 4]
    dt = dt,                    # real[N_intervals]
    lemma_freq = lemma_freq_std, # real[N_intervals]
    prop_bipartite = prop_bipartite,
    lemma_id = lemma_id,
    variety = variety,
    principal_part = principal_part,
    N_lemmas = length(lemma_levels),
    N_varieties = max(variety, na.rm = TRUE),
    N_corpora = max(corpus, na.rm = TRUE),
    alpha_prior = rep(1.0, 4)   # Dirichlet prior for state probs
  )

  return(stan_data)
}


# Modify run_analysis to use the new preparation function when `transitions = TRUE`
run_analysis <- function(obs_file, seq_file, model_file, transitions = TRUE, lemma_freq_column = "avg_lemma_freq", n_sample = NA, output_file = NA){
  if (is.na(output_file)) {
    file_name <- basename(model_file)
    file_name <- strsplit(file_name, split=".", fixed=TRUE)[[1]][1]
    output_file <- paste0("analysis/results/", file_name, "_fit.rds")
  }
  if (!dir.exists("analysis/results")) dir.create("analysis/results", recursive = TRUE)

  if (file.exists(output_file)) {
    overwrite <- readline(prompt = paste("Output file", output_file, "already exists. Overwrite? (y/n): "))
    if (tolower(overwrite) != "y") stop("Exiting without overwriting existing file.")
    file.remove(output_file)
  }

  if (transitions) {
    stan_data <- prepare_stan_data_transitions(obs_file, seq_file, lemma_freq_column = lemma_freq_column, n_sample = n_sample)
  } else {
    stan_data <- prepare_stan_data_aggregated(obs_file, seq_file, lemma_freq_column = lemma_freq_column, n_sample = n_sample)
  }

  # Compile and run model
  model <- stan_model(file = model_file)
  fit <- sampling(
    model,
    data = stan_data,
    iter = 2000,
    chains = 4,
    warmup = 1000,
    seed = 97,
    refresh = 100
  )

  saveRDS(fit, output_file)
  return(fit)
}

# Allow running from command line with arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 3) {
  obs_file <- args[1]
  seq_file <- args[2]
  model_file <- args[3]
  transitions <- if (length(args) >= 4) as.logical(args[4]) else TRUE
  lemma_freq_column <- if (length(args) >= 5) args[5] else "avg_lemma_freq"
  n_sample <- if (length(args) >= 6) {
    v <- args[6]
    if (tolower(v) %in% c("na", "none", "")) NA else as.integer(v)
  } else NA
  output_file <- if (length(args) >= 7) args[7] else NA

  fit <- run_analysis(
    obs_file = obs_file,
    seq_file = seq_file,
    model_file = model_file,
    transitions = transitions,
    lemma_freq_column = lemma_freq_column,
    n_sample = n_sample,
    output_file = output_file
    )
}