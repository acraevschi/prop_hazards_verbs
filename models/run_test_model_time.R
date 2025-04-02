library(rstan)
library(dplyr)
library(splines)  # For B-spline basis generation

# Read and prepare data
data_path <- "data/verner_transitions.csv"
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

# Generate B-spline basis matrix for temporal emissions | this is a placeholder, we can think of concrete basis and degree later
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
  N_verbs = max(data$form_id),
  N_obs = nrow(data),
  N_states = 4,
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
  # Spline-related data
  num_basis = num_basis,
  basis = basis
)

# Run Stan model
hmm_fit <- stan(
  file = "models/stan_models/test_model_time.stan",
  data = data_list,
  iter = 2500,
  warmup = 1500,
  chains = 4,
  cores = 4,
)

print(hmm_fit)
