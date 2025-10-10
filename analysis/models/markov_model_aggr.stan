functions {
  matrix get_transition_matrix(vector rates, real beta_form, real beta_lemma,
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
          real base = rates[idx];
          real lr = log(base + 1e-9)
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
    if (time_interval <= 0) P = diag_matrix(rep_vector(1.0, 4));
    else P = matrix_exp(Q * time_interval);
    return P;
  }
}

data {
  int<lower=1> M; // sequences
  int<lower=1> max_T;
  int<lower=1> T_len[M];

  int<lower=0> counts[M, max_T, 4]; // aggregated counts per timepoint
  int<lower=0> totals[M, max_T];    // totals (sum of counts) optional / for checks
  real time_intervals[M, max_T - 1];

  real form_freq[M, max_T];   // log already
  real lemma_freq[M, max_T];  // log already
  real prop_bipartite[M, max_T]; // [0..1], lemma-level at source time

  int<lower=1> N_lemmas;
  int<lower=1> N_varieties;
  int<lower=1> N_corpora;
  int<lower=1,upper=N_lemmas> lemma_id[M];
  int<lower=1,upper=4> principal_part[M];

  vector[4] alpha_prior; // Dirichlet prior for initial pi
}

parameters {
  vector<lower=0>[12] baseline_rates; // off-diagonal base rates
  real beta_form;
  real beta_lemma;
  real beta_bipartite;

  vector[N_lemmas] lemma_eff_raw;
  real<lower=0> sigma_lemma;

  simplex[4] pi_init[M]; // initial distribution for each sequence
}

transformed parameters {
  vector[N_lemmas] lemma_eff = sigma_lemma * lemma_eff_raw;
}

model {
  // Priors
  baseline_rates ~ exponential(1.0);
  beta_form ~ normal(0, 1);
  beta_lemma ~ normal(0, 1);
  beta_bipartite ~ normal(0, 1);
  lemma_eff_raw ~ normal(0,1);
  sigma_lemma ~ exponential(1.0);

  // prior for initial distributions
  for (m in 1:M) {
    target += dirichlet_lpdf(pi_init[m] | alpha_prior);
  }

  // Likelihood: multinomial observations informed by CTMC propagation of π
  for (m in 1:M) {
    // initial timepoint: counts at t=1 inform pi_init[m]
    if (T_len[m] >= 1) {
      counts[m,1] ~ multinomial(pi_init[m]);
    }

    // propagate forward deterministically with the CTMC, then compare to observed counts
    row_vector[4] pi = to_row_vector(pi_init[m]); // pi at time t
    for (t in 1:(T_len[m] - 1)) {
      matrix[4,4] P = get_transition_matrix(
                        baseline_rates, beta_form, beta_lemma, beta_bipartite,
                        lemma_eff[lemma_id[m]],
                        form_freq[m,t], lemma_freq[m,t],
                        prop_bipartite[m,t], time_intervals[m,t]);
      pi = pi * P; // next distribution
      counts[m, t+1] ~ multinomial(to_vector(pi));
    }
  }
}

generated quantities {
  vector[M] log_lik_seq;
  for (m in 1:M) {
    real lp = 0;
    // initial
    if (T_len[m] >= 1) {
      lp += multinomial_lpmf(counts[m,1] | pi_init[m]);
    }
    // forward
    row_vector[4] pi = to_row_vector(pi_init[m]);
    for (t in 1:(T_len[m] - 1)) {
      matrix[4,4] P = get_transition_matrix(
                        baseline_rates, beta_form, beta_lemma, beta_bipartite,
                        sigma_lemma * lemma_eff_raw[lemma_id[m]],
                        form_freq[m,t], lemma_freq[m,t],
                        prop_bipartite[m,t], time_intervals[m,t]);
      pi = pi * P;
      lp += multinomial_lpmf(counts[m, t+1] | to_vector(pi));
    }
    log_lik_seq[m] = lp;
  }
}
