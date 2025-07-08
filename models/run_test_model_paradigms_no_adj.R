library(rstan)
library(dplyr)
library(splines)  # For B-spline basis generation
library(jsonlite)  # For reading JSON metadata

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Read and prepare data
data_path <- "data/transitions_data.csv"
metadata_path <- "data/transitions_data_metadata.json"

if (!file.exists(data_path)) {
    setwd("..")  # Move up one level if running from the `models/stan_models/` folder
}

data <- read.csv(data_path) %>%
  arrange(form_id, time) %>%
  mutate(
    form_id = as.integer(factor(form_id)),
    dialect_id = as.integer(factor(dialect_group)),
    # Add synthetic frequency just for testing purposes
    freq = exp(rnorm(nrow(.), mean = 0, sd = 0.75))
  )

# Read metadata with paradigm and irregularity information
metadata <- fromJSON(metadata_path)

# Generate B-spline basis matrix for temporal emissions 
num_basis <- 5  # Start with 5 basis functions 
basis <- bs(data$time, 
           df = num_basis, 
           degree = 3,  # Cubic splines
           intercept = TRUE)

# Create verb indices
verb_indices <- data %>%
  group_by(form_id) %>%
  summarise(
    start = min(row_number()),
    end = max(row_number()),
    .groups = 'drop'
  )

# Prepare Stan data
data_list <- list(
  # Original variables
  N_verbs = max(data$form_id),
  N_obs = nrow(data),
  N_states = metadata$n_principal_parts,
  form = data$form_id,
  time = data$time,
  time_since_prev = data$time_since_prev/100,  # Scaling if needed
  obs_v = data$obs_v,
  obs_c = data$obs_c,
  freq = data$freq,
  dialect_id = data$dialect_id,
  n_dialects = max(data$dialect_id),
  verb_starts = verb_indices$start,
  verb_ends = verb_indices$end,
  num_basis = num_basis,
  basis = basis,
  # New variables for paradigm model
  N_lemmas = metadata$n_lemmas,
  lemma_id = data$lemma_id,
  # N_principal_parts = metadata$n_principal_parts, # Will be corrected later in python script
  N_principal_parts = metadata$n_principal_parts,
  principal_part_id = data$principal_part_id,
  n_time_points = metadata$n_time_points,
  unique_times = metadata$unique_times,
  irregularity_index = metadata$irregularity_index,
  m_values = metadata$m_values
)

# Run Stan model
hmm_fit <- stan(
  file = "models/stan_models/test_model_paradigms.stan",
  data = data_list,
  iter = 2500,
  warmup = 1500,
  chains = 4,
  cores = 4,
  seed = 97,
  control=list(adapt_delta=0.975)
)

output_dir <- "fits/"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_path <- paste0(output_dir, "test_fit_paradigms.rds")

if (file.exists(output_path)) {
  file.remove(output_path)
}

saveRDS(hmm_fit, output_path)
# rows with Rhat == NaN are fine, those are just the parameters that are not estimated in this model
