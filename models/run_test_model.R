library(rstan)
library(dplyr)

### TODO: Explain and make it more functional and readable 

# Read the CSV file
if (!file.exists("data/verner_transitions.csv")) {
    setwd("..")  # Move up one level if running from the `models/stan_models/` folder
}
data <- read.csv("data/verner_transitions.csv")

# assign random freq for testing
set.seed(123)
data$freq <- exp(rnorm(nrow(data), mean = 0, sd = 0.75))

data <- data %>%
  arrange(verb, time) %>%
  mutate(verb_id = as.integer(factor(verb)))

# Create observation indices for each verb
verb_indices <- data %>%
  group_by(verb_id) %>%
  summarise(
    start = min(row_number()),
    end = max(row_number()),
    .groups = 'drop'
  )

# Convert to list format for Stan
data_list <- list(
  N_verbs = max(data$verb_id),
  N_obs = nrow(data),
  N_states = 4,
  verb = data$verb_id,
  time = data$time,
  time_since_prev = data$time_since_prev/100, # to make it per 100 years
  obs_v = data$obs_v,
  obs_c = data$obs_c,
  freq = data$freq,
  
  # Additional indices for efficient verb-specific access
  verb_starts = verb_indices$start,
  verb_ends = verb_indices$end
)

hmm_fit <- stan(
  file = "models/stan_models/test_model.stan",
  data = data_list,
  iter = 2500,
  warmup = 1500,
  chains = 4,
  cores = 4,
)

print(hmm_fit)
