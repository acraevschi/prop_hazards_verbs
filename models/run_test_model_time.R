library(rstan)
library(dplyr)
library(splines)  # For B-spline basis generation

# Read and prepare data
if (!file.exists("data/verner_transitions.csv")) {
    setwd("..")  # Move up one level if running from the `models/stan_models/` folder
}

data <- read.csv("data/verner_transitions.csv") %>%
  arrange(verb, time) %>%
  mutate(
    verb_id = as.integer(factor(verb)),
    # Add synthetic frequency if needed
    freq = exp(rnorm(nrow(.), mean = 0, sd = 0.75))
  )

# Generate B-spline basis matrix for temporal emissions
num_basis <- 5  # Start with 5 basis functions 
basis <- bs(data$time, 
           df = num_basis, 
           degree = 3,  # Cubic splines
           intercept = TRUE)

# Create verb indices
verb_indices <- data %>%
  group_by(verb_id) %>%
  summarise(
    start = min(row_number()),
    end = max(row_number()),
    .groups = 'drop'
  )

# Prepare Stan data
data_list <- list(
  N_verbs = max(data$verb_id),
  N_obs = nrow(data),
  N_states = 4,
  verb = data$verb_id,
  time = data$time,
  time_since_prev = data$time_since_prev/100,  # Scaling if needed
  obs_v = data$obs_v,
  obs_c = data$obs_c,
  freq = data$freq,
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
