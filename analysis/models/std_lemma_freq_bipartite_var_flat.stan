functions {
  // Build transition probability matrix P = expm(Q * dt)
  matrix get_transition_matrix(vector log_rates, real beta_lemma,
                               real beta_bipartite,
                               real lemma_eff,
                               real lemma_freq,
                               real var_effect,
                               real prop_bipartite, real time_interval) {
    matrix[4,4] Q;
    matrix[4,4] P;
    Q = rep_matrix(0.0, 4, 4);
    int idx = 1;
    for (i in 1:4) {
      for (j in 1:4) {
        if (i != j) {
          real lr = log_rates[idx]
                    + beta_lemma * lemma_freq
                    + lemma_eff + var_effect;
          if (i == 4) {
            lr += beta_bipartite * prop_bipartite;
          }
          Q[i,j] = exp(lr);
          idx += 1;
        }
      }
    }
    for (i in 1:4) Q[i,i] = -sum(Q[i,]);
    if (time_interval == 0) P = diag_matrix(rep_vector(1.0, 4));
    else P = matrix_exp(Q * time_interval);
    return P;
  }
}

data {
  int<lower=1> N_intervals;

  int<lower=0> counts_t[N_intervals, 4];
  int<lower=0> counts_tp1[N_intervals, 4];
  real dt[N_intervals];

  real lemma_freq[N_intervals];     // standardized
  real prop_bipartite[N_intervals]; // [0..1]
  int<lower=1> lemma_id[N_intervals];
  int<lower=1> variety[N_intervals];
  int<lower=1> principal_part[N_intervals];

  int<lower=1> N_lemmas;
  int<lower=1> N_varieties;

  vector[4] alpha_prior; // Dirichlet prior for initial pi (when you want to treat counts_t as prior+obs)
}

parameters {
  vector[12] log_baseline_rates; // log of off-diagonal base rates
  real beta_lemma;
  real beta_bipartite;

  vector[N_lemmas] lemma_eff_raw;
  real<lower=0> sigma_lemma;

  vector[N_varieties] var_effect_raw;
  real<lower=0> sigma_var;

  real<lower=0> kappa; // concentration for converting propagated mean -> Dirichlet prior
}

transformed parameters {
  vector[N_lemmas] lemma_eff = sigma_lemma * lemma_eff_raw;
  vector[N_varieties] var_effect = var_effect_raw * sigma_var;
}

model {
  // Priors
  log_baseline_rates ~ normal(0, 1);
  beta_lemma ~ normal(0, 1);
  beta_bipartite ~ normal(0, 1);
  lemma_eff_raw ~ normal(0,1);
  sigma_lemma ~ exponential(3);
  var_effect_raw ~ normal(0,1);
  sigma_var ~ exponential(3);
  kappa ~ gamma(3, 0.5);

  // Likelihood: loop over intervals
  for (n in 1:N_intervals) {
    // posterior Dirichlet after seeing counts at time t
    vector[4] alpha = alpha_prior + to_vector(counts_t[n]);

    // posterior mean at time t
    real alpha_sum = sum(alpha);
    vector[4] mu = alpha / alpha_sum;

    // propagate mean through CTMC for interval dt[n]
    matrix[4,4] P = get_transition_matrix(
                      log_baseline_rates, beta_lemma, beta_bipartite,
                      lemma_eff[lemma_id[n]],
                      lemma_freq[n],
                      var_effect[variety[n]],
                      prop_bipartite[n], dt[n]);

    row_vector[4] mu_row = to_row_vector(mu);
    row_vector[4] mu_pred_row = mu_row * P;
    vector[4] mu_pred = to_vector(mu_pred_row);

    // convert propagated mean to Dirichlet prior (concentration kappa)
    vector[4] alpha_pred = kappa * mu_pred;

    // likelihood for counts at t+1 marginalized over pi_{t+1}
    int N_tot = sum(counts_tp1[n]);
    target += lgamma(sum(alpha_pred)) - lgamma(sum(alpha_pred) + N_tot);
    for (k in 1:4) {
        target += lgamma(alpha_pred[k] + counts_tp1[n, k]) - lgamma(alpha_pred[k]);
    }
  }
}

generated quantities {
  // Optionally compute per-interval log-likelihoods
  vector[N_intervals] log_lik_interval;
  for (n in 1:N_intervals) {
    vector[4] alpha = alpha_prior + to_vector(counts_t[n]);
    real alpha_sum = sum(alpha);
    vector[4] mu = alpha / alpha_sum;

    matrix[4,4] P = get_transition_matrix(
                      log_baseline_rates, beta_lemma, beta_bipartite,
                      lemma_eff[lemma_id[n]],
                      lemma_freq[n],
                      var_effect[variety[n]],
                      prop_bipartite[n], dt[n]);
    row_vector[4] mu_row = to_row_vector(mu);
    row_vector[4] mu_pred_row = mu_row * P;
    vector[4] mu_pred = to_vector(mu_pred_row);
    vector[4] alpha_pred = kappa * mu_pred;

    int N_tot = sum(counts_tp1[n]);
    real ll = lgamma(sum(alpha_pred)) - lgamma(sum(alpha_pred) + N_tot);
    for (k in 1:4) {
        ll += lgamma(alpha_pred[k] + counts_tp1[n, k]) - lgamma(alpha_pred[k]);
    }
    log_lik_interval[n] = ll;

  }
}
