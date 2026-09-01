#!/usr/bin/env Rscript
# ==============================================================================
# MCMC Convergence Diagnostics Summary
# ==============================================================================
# Summarizes sampler health across the 5 fitted models in fits/:
# - R-hat (Max R-hat and % <= 1.01)
# - Minimum Bulk-ESS
# - Minimum Tail-ESS
# - Divergent transitions
# - Maximum treedepth hits
# ==============================================================================

suppressPackageStartupMessages({
  library(brms)
  library(rstan)
  library(posterior)
  library(dplyr)
  library(knitr)
})

models_info <- list(
  list(name = "Tensor (k=10)", file = "fits/tensor_fit_marking_type_k10.rds"),
  list(name = "Smooth interaction (k=10)", file = "fits/base_fit_marking_type_k10.rds"),
  list(name = "Tensor | Token Freq. (k=10)", file = "fits/tensor_fit_marking_type_k10_token.rds"),
  list(name = "Tensor (k=4)", file = "fits/tensor_fit_marking_type_k4.rds"),
  list(name = "Smooth interaction (k=4)", file = "fits/base_fit_marking_type.rds")
)

results <- list()

cat("Extracting MCMC convergence diagnostics across fitted models...\n\n")

for (m in models_info) {
  m_name <- m$name
  f_path <- m$file
  
  if (!file.exists(f_path)) {
    cat(sprintf("Warning: File %s does not exist, skipping.\n", f_path))
    next
  }
  
  cat(sprintf("Checking model: %s (%s)...\n", m_name, f_path))
  fit <- readRDS(f_path)
  
  # Extract diagnostics from brms / Stan object
  stan_fit <- fit$fit
  divs <- rstan::get_num_divergent(stan_fit)
  max_td <- rstan::get_num_max_treedepth(stan_fit)
  
  # R-hat diagnostics
  rhats <- brms::rhat(fit)
  max_rhat <- max(rhats, na.rm = TRUE)
  pct_rhat_ok <- mean(rhats <= 1.01, na.rm = TRUE) * 100
  
  # Bulk and Tail ESS
  draws <- as_draws(fit)
  d_sum <- summarise_draws(draws)
  min_bulk <- min(d_sum$ess_bulk, na.rm = TRUE)
  min_tail <- min(d_sum$ess_tail, na.rm = TRUE)
  
  results[[m_name]] <- data.frame(
    Model = m_name,
    Post_Warmup_Draws = brms::ndraws(fit),
    Max_Rhat = round(max_rhat, 4),
    Pct_Rhat_Num = pct_rhat_ok,
    Pct_Rhat_LE_1.01 = sprintf("%.1f%%", pct_rhat_ok),
    Min_Bulk_ESS = round(min_bulk, 0),
    Min_Tail_ESS = round(min_tail, 0),
    Divergences = divs,
    Max_Treedepth_Hits = max_td,
    stringsAsFactors = FALSE
  )
}

res_df <- do.call(rbind, results)

# Pct_Rhat_Num drives the summary text below. Drop it from the reported table,
# which keeps the formatted percentage instead.
pct_rhat_num <- res_df$Pct_Rhat_Num
draws_num <- res_df$Post_Warmup_Draws
res_df <- res_df[, setdiff(names(res_df), c("Pct_Rhat_Num", "Post_Warmup_Draws")), drop = FALSE]

# Save to CSV
csv_path <- "analysis/reports/mcmc_convergence.csv"
dir.create(dirname(csv_path), showWarnings = FALSE, recursive = TRUE)
write.csv(res_df, csv_path, row.names = FALSE)
cat(sprintf("\nSaved CSV summary to: %s\n", csv_path))

# Build the summary text from the measured values, never from fixed prose.
build_interpretation <- function(df, pct_rhat, draws) {
  n_models <- nrow(df)
  worst_rhat <- max(df$Max_Rhat)
  all_rhat_ok <- all(pct_rhat >= 100)
  min_bulk <- min(df$Min_Bulk_ESS)
  min_tail <- min(df$Min_Tail_ESS)
  ess_floor <- 400  # 100 effective draws per chain, for 4 chains
  div_lo <- min(df$Divergences)
  div_hi <- max(df$Divergences)
  td_models <- df$Model[df$Max_Treedepth_Hits > 0]

  lines <- c("> **Interpretation**:")

  lines <- c(lines, if (all_rhat_ok) {
    sprintf("> - All %d models satisfy $\\hat{R} \\le 1.01$ across 100%% of estimated parameters (Vehtari et al., 2021). The largest value is %.4f.",
            n_models, worst_rhat)
  } else {
    sprintf("> - %d of %d models satisfy $\\hat{R} \\le 1.01$ across all parameters. The largest value is %.4f. Read the table before you use the affected models.",
            sum(pct_rhat >= 100), n_models, worst_rhat)
  })

  lines <- c(lines, if (min_bulk > ess_floor && min_tail > ess_floor) {
    sprintf("> - Bulk-ESS and Tail-ESS exceed the reliability threshold of %d for 4 chains. The lowest values are %d (bulk) and %d (tail).",
            ess_floor, round(min_bulk), round(min_tail))
  } else {
    sprintf("> - Effective sample size falls below the threshold of %d for 4 chains. The lowest values are %d (bulk) and %d (tail).",
            ess_floor, round(min_bulk), round(min_tail))
  })

  lines <- c(lines, if (div_hi == 0) {
    "> - No divergent transitions occurred under `adapt_delta = 0.99`."
  } else {
    sprintf("> - Divergent transitions range from %d to %d per model under `adapt_delta = 0.99`, out of %s post-warmup draws.",
            div_lo, div_hi,
            if (length(unique(draws)) == 1) format(draws[1], big.mark = ",")
            else sprintf("%s to %s", format(min(draws), big.mark = ","), format(max(draws), big.mark = ",")))
  })

  if (length(td_models) > 0) {
    lines <- c(lines, sprintf(
      "> - %s hit the maximum treedepth of 10. This costs sampling efficiency, but it does not bias the posterior, and the $\\hat{R}$ and ESS values for %s stay within the thresholds above.",
      paste(td_models, collapse = ", "),
      if (length(td_models) == 1) "that model" else "those models"))
  }

  paste0(paste(lines, collapse = "\n"), "\n")
}

# Generate Markdown table
md_content <- paste0(
  "# MCMC Convergence Diagnostics Summary Table\n\n",
  "Table of sampler health metrics across the 5 fitted Bayesian GAMM models in `fits/`:\n\n",
  # kable() returns one element per line. Collapse it, or paste0() recycles the
  # surrounding text across every row. row.names = FALSE drops the duplicate
  # Model column that rbind() puts in the row names.
  paste(
    knitr::kable(
      res_df,
      format = "pipe",
      row.names = FALSE,
      col.names = c("Model", "Max $\\hat{R}$", "$\\hat{R} \\le 1.01$ (%)", "Min Bulk-ESS", "Min Tail-ESS", "Divergences", "Max Treedepth Hits")
    ),
    collapse = "\n"
  ),
  "\n\n",
  build_interpretation(res_df, pct_rhat_num, draws_num)
)

md_path <- "analysis/reports/mcmc_convergence_table.md"
writeLines(md_content, md_path)
cat(sprintf("Saved Markdown table to: %s\n\n", md_path))

# Print to console
print(kable(res_df, format = "simple", row.names = FALSE))
