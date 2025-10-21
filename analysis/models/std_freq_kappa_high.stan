functions {
  // Build transition probability matrix P = expm(Q * dt)
  matrix get_transition_matrix(vector log_rates, real beta_form, real beta_lemma,
                               real beta_bipartite,
                               real lemma_eff,
                               real form_freq, real lemma_freq,
                               real prop_bipartite, real time_interval) {
    matrix[4,4] Q;
    matrix[4,4] P;
    Q = rep_matrix(0.0, 4, 4);
    int idx = 1;
    for (i in 1:4) {
      for (j in 1:4) {
        if (i != j) {
          real lr = log_rates[idx]
                    + beta_form * form_freq
                    + beta_lemma * lemma_freq
                    + beta_bipartite * prop_bipartite
                    + lemma_eff;
          Q[i,j] = exp(lr);
          idx += 1;
        }
      }
    }
    for (i in 1:4) Q[i,i] = -sum(Q[i,]);
    if (time_interval <= 0)
      P = diag_matrix(rep_vector(1.0, 4));
    else
      P = matrix_exp(Q * time_interval);
    return P;
  }
}

data {
  int<lower=1> M; // sequences
  int<lower=1> max_T;
  int<lower=1> T_len[M];

  int<lower=0> counts[M, max_T, 4]; // aggregated counts per timepoint
  int<lower=0> totals[M, max_T];    // totals per timepoint (for checks)
  real time_intervals[M, max_T - 1];

  real form_freq[M, max_T];   // log-transformed already
  real lemma_freq[M, max_T];  // log-transformed already
  real prop_bipartite[M, max_T]; // [0..1], lemma-level at source time

  int<lower=1> N_lemmas;
  int<lower=1> N_varieties;
  int<lower=1> N_corpora;
  int<lower=1,upper=N_lemmas> lemma_id[M];
  int<lower=1,upper=4> principal_part[M];

  vector[4] alpha_prior; // Dirichlet prior for initial pi
}

parameters {
  vector[12] log_baseline_rates; // off-diagonal base log-rates
  real beta_form;
  real beta_lemma;
  real beta_bipartite;

  vector[N_lemmas] lemma_eff_raw;
  real<lower=0> sigma_lemma;

  real<lower=0> kappa; // concentration for converting propagated mean -> Dirichlet prior
}

transformed parameters {
  vector[N_lemmas] lemma_eff = sigma_lemma * lemma_eff_raw;
}

model {
  // Priors
  log_baseline_rates ~ normal(0, 1);
  beta_form ~ normal(0, 1);
  beta_lemma ~ normal(0, 1);
  beta_bipartite ~ normal(0, 1);
  lemma_eff_raw ~ normal(0, 1);
  sigma_lemma ~ exponential(3.0);
  kappa ~ gamma(3, 0.5);

  // Likelihood via sequential Bayesian updating (Dirichlet prior + CTMC propagation)
  for (m in 1:M) {
    vector[4] alpha = alpha_prior;

    // initial observation
    if (T_len[m] >= 1) {
      target += lgamma(sum(alpha)) - lgamma(sum(counts[m,1]) + sum(alpha))
                + sum(lgamma(to_vector(counts[m,1]) + alpha) - lgamma(alpha));
      for (k in 1:4)
        alpha[k] = alpha[k] + counts[m,1,k];
    }

    // propagate and update
    for (t in 1:(T_len[m] - 1)) {
      matrix[4,4] P = get_transition_matrix(
                        log_baseline_rates, beta_form, beta_lemma, beta_bipartite,
                        lemma_eff[lemma_id[m]],
                        form_freq[m,t], lemma_freq[m,t],
                        prop_bipartite[m,t], time_intervals[m,t]);

      real alpha_sum = sum(alpha);
      vector[4] mu = alpha / alpha_sum;
      row_vector[4] mu_pred_row = to_row_vector(mu) * P;
      vector[4] mu_pred = to_vector(mu_pred_row);
      vector[4] alpha_pred = kappa * mu_pred;

      target += lgamma(sum(alpha_pred)) - lgamma(sum(counts[m,t+1]) + sum(alpha_pred))
                + sum(lgamma(to_vector(counts[m,t+1]) + alpha_pred) - lgamma(alpha_pred));

      for (k in 1:4)
        alpha[k] = alpha_pred[k] + counts[m,t+1,k];
    }
  }
}

generated quantities {
  vector[M] log_lik_seq;
  for (m in 1:M) {
    real lp = 0;
    vector[4] alpha = alpha_prior;

    if (T_len[m] >= 1) {
      lp += lgamma(sum(alpha)) - lgamma(sum(counts[m,1]) + sum(alpha))
            + sum(lgamma(to_vector(counts[m,1]) + alpha) - lgamma(alpha));
      for (k in 1:4)
        alpha[k] = alpha[k] + counts[m,1,k];
    }

    for (t in 1:(T_len[m] - 1)) {
      matrix[4,4] P = get_transition_matrix(
                        log_baseline_rates, beta_form, beta_lemma, beta_bipartite,
                        lemma_eff[lemma_id[m]],
                        form_freq[m,t], lemma_freq[m,t],
                        prop_bipartite[m,t], time_intervals[m,t]);

      real alpha_sum = sum(alpha);
      vector[4] mu = alpha / alpha_sum;
      row_vector[4] mu_pred_row = to_row_vector(mu) * P;
      vector[4] mu_pred = to_vector(mu_pred_row);
      vector[4] alpha_pred = kappa * mu_pred;

      lp += lgamma(sum(alpha_pred)) - lgamma(sum(counts[m,t+1]) + sum(alpha_pred))
            + sum(lgamma(to_vector(counts[m,t+1]) + alpha_pred) - lgamma(alpha_pred));

      for (k in 1:4)
        alpha[k] = alpha_pred[k] + counts[m,t+1,k];
    }

    log_lik_seq[m] = lp;
  }
}
