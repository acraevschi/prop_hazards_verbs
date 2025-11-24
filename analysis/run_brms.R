library(dplyr)
library(brms)


model_data <- read.csv("analysis/pauls_principle_dataset.csv")
model_data <- unique(model_data)
model_data$is_bipartite <- as.factor(model_data$is_bipartite)

####### MODEL ########

# 1. Model formula
# ------------------------------------
hierarchical_model_formula <- bf(
  has_levelled ~ is_bipartite + dialect + s(date) + 
                + (1 | lemma_std) + log_freq,
  family = bernoulli()
)

hierarchical_model_interaction_formula <- bf(
  has_levelled ~ is_bipartite + s(date, by = is_bipartite) + 
                 dialect + log_freq + (1 | lemma_std),
  family = bernoulli()
)

# 2. Run the hierarchical and interaction model
# ------------------------------------
hierarchical_model_fit <- brm(
  formula = hierarchical_model_formula,
  data = model_data,
  chains = 4,
  iter = 2000,
  cores = 4,
  backend = "rstan",
  control = list(adapt_delta = 0.99),
  file = "analysis/models/hierarchical_brms_fit"
)

hierarchical_model_interaction_fit <- brm(
  formula = hierarchical_model_interaction_formula,
  data = model_data,
  chains = 4,
  iter = 2000,
  cores = 4,
  backend = "rstan",
  control = list(adapt_delta = 0.99),
  file = "analysis/models/hierarchical_interaction_brms_fit"
)


# 3. View the results
# ------------------------------------
print(summary(hierarchical_model_fit))

print(summary(hierarchical_model_interaction_fit))

# 4. Plot the fixed effects (like 'is_bipartite') and the spline
# ------------------------------------
plot(conditional_effects(hierarchical_model_fit), points = TRUE)

plot(conditional_effects(hierarchical_model_interaction_fit), points = TRUE)