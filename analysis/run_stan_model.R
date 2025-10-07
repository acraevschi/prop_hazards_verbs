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

# Run Stan model (rest of the function remains the same)
run_analysis <- function(obs_file, seq_file, model_file, output_file = NA) {
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
  stan_data <- prepare_stan_data(obs_file, seq_file)
  
  # Compile and run model
  model <- stan_model(file = model_file)
  
  # Run sampling
  fit <- sampling(
    model,
    data = stan_data,
    iter = 2000,
    chains = 1,
    warmup = 1000,
    seed = 97
  )
  
  # Save results
  saveRDS(fit, output_file)
  
  return(fit)
}

fit <- run_analysis(
    obs_file = "analysis/obs.csv",
    seq_file = "analysis/seq.csv", 
    model_file = "analysis/models/sequential_hmm.stan"
  )
