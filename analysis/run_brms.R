library(dplyr)
library(brms)

model_data <- read.csv("analysis/pauls_principle_dataset.csv")
model_data <- model_data %>%
  distinct(across(-c(form, inflcat)), .keep_all = TRUE)

model_data$is_bipartite <- as.factor(model_data$is_bipartite)
model_data$element_type <- as.factor(model_data$element_type)

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

levelling_type_formula <- bf(
  has_levelled ~ 
    is_bipartite + 
    element_type +
    s(date, by = element_type) + # The trajectory of change over time, split by Element Type (vowel or consonant)
    dialect + 
    log_freq + 
    (1 | lemma_std),
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

levelling_type_model_fit <- brm(
  formula = levelling_type_formula,
  data = model_data,
  chains = 4,
  iter = 2000,
  cores = 4,
  backend = "rstan",
  control = list(adapt_delta = 0.99),
  file = "analysis/models/levelling_type_brms_fit"
)

# 3. View the results
# ------------------------------------
print(summary(hierarchical_model_fit))

print(summary(hierarchical_model_interaction_fit))

print(summary(levelling_type_model_fit))

# 4. Plot the fixed effects (like 'is_bipartite') and the spline
# ------------------------------------
plot(conditional_effects(hierarchical_model_fit), points = TRUE)

plot(conditional_effects(hierarchical_model_interaction_fit), points = TRUE)

plot(conditional_effects(levelling_type_model_fit), points = TRUE)
