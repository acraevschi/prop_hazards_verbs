functions {
  matrix get_transition_matrix(vector rates, real beta_form, real beta_lemma,
                               vector beta_bipartite, real variety_eff,
                               real corpus_eff, real lemma_eff,
                               real form_freq, real lemma_freq,
                               int is_bipartite, real time_interval) {
    matrix[4,4] Q;
    matrix[4,4] P;
    Q = rep_matrix(0, 4, 4);
    int idx = 1;
    for (i in 1:4) for (j in 1:4) if (i != j) {
      real base = rates[idx];
      real lr = log(base + 1e-12) + beta_form * form_freq + beta_lemma * lemma_freq + beta_bipartite[i] + variety_eff + corpus_eff + lemma_eff;
      Q[i,j] = exp(lr);
      idx += 1;
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
  int<lower=1,upper=4> obs_states[M, max_T];
  real time_intervals[M, max_T-1];
  real form_freq[M, max_T];
  real lemma_freq[M, max_T];

  int<lower=1> N_lemmas;
  int<lower=1> N_varieties;
  int<lower=1> N_corpora;
  int<lower=1,upper=N_lemmas> lemma_id[M];
  int<lower=1,upper=4> principal_part[M];
  int<lower=1,upper=N_varieties> variety[M];
  int<lower=1,upper=N_corpora> corpus[M];
  vector[4] alpha_prior; // prior for initial latent distribution per lemma (Dirichlet)
}

parameters {
  vector<lower=0>[12] baseline_rates;
  real beta_form;
  real beta_lemma;
  vector[3] beta_bipartite_raw;

  vector[N_varieties] variety_eff_raw;
  vector[N_corpora] corpus_eff_raw;
  vector[N_lemmas] lemma_eff_raw;
  real<lower=0> sigma_variety;
  real<lower=0> sigma_corpus;
  real<lower=0> sigma_lemma;

  // initial latent distribution per lemma
  simplex[4] pi_lemma[N_lemmas];

  // emission parameters
  vector[4] emission_logit_base;
  vector[N_varieties] emission_variety_raw;
  vector[N_corpora] emission_corpus_raw;
  vector[4] emission_pp_raw;
  real<lower=0> sigma_em_var;
  real<lower=0> sigma_em_corp;
  real<lower=0> sigma_em_pp;
  simplex[3] emission_off[4];
  real beta_emiss_form;
  real beta_emiss_logtime;
}

transformed parameters {
  vector[N_varieties] variety_eff = sigma_variety * variety_eff_raw;
  vector[N_corpora] corpus_eff = sigma_corpus * corpus_eff_raw;
  vector[N_lemmas] lemma_eff = sigma_lemma * lemma_eff_raw;
  vector[4] beta_bipartite;
  beta_bipartite[1:3] = beta_bipartite_raw;
  beta_bipartite[4] = 0;

  vector[N_varieties] emission_variety = sigma_em_var * emission_variety_raw;
  vector[N_corpora] emission_corpus = sigma_em_corp * emission_corpus_raw;
  vector[4] emission_pp = sigma_em_pp * emission_pp_raw;
  vector[M] log_lik_m;

  // forward algorithm per sequence - moved from model block
  for (m in 1:M) {
    int Tm = T_len[m];
    vector[4] log_alpha;

    // initial alpha at t=1
    for (z in 1:4) {
      real diag_logit = emission_logit_base[z] + emission_variety[variety[m]] + emission_corpus[corpus[m]] + emission_pp[principal_part[m]] + beta_emiss_form * form_freq[m,1] + beta_emiss_logtime * log1p(time_intervals[m,1]);
      real diag_p = inv_logit(diag_logit);
      vector[3] off = emission_off[z];
      real p_obs;
      int kk = 1;
      if (obs_states[m,1] == z) p_obs = diag_p;
      else {
        for (o in 1:4) if (o != z) {
          if (o == obs_states[m,1]) p_obs = (1 - diag_p) * off[kk];
          kk += 1;
        }
      }
      log_alpha[z] = log(pi_lemma[lemma_id[m]][z]) + log(fmax(p_obs, 1e-12));
    }

    // iterate t=2..Tm
    for (t in 2:Tm) {
      matrix[4,4] P = get_transition_matrix(baseline_rates, beta_form, beta_lemma, beta_bipartite,
                                            variety_eff[variety[m]], corpus_eff[corpus[m]], lemma_eff[lemma_id[m]],
                                            form_freq[m,t-1], lemma_freq[m,t-1], (obs_states[m,t-1]==4) ? 1 : 0, time_intervals[m,t-1]);
      vector[4] e;
      for (z in 1:4) {
        real diag_logit = emission_logit_base[z] + emission_variety[variety[m]] + emission_corpus[corpus[m]] + emission_pp[principal_part[m]] + beta_emiss_form * form_freq[m,t] + beta_emiss_logtime * log1p(time_intervals[m,t-1]);
        real diag_p = inv_logit(diag_logit);
        int kk = 1;
        if (obs_states[m,t] == z) e[z] = diag_p;
        else {
          for (o in 1:4) if (o != z) {
            if (o == obs_states[m,t]) e[z] = (1 - diag_p) * emission_off[z][kk];
            kk += 1;
          }
        }
      }
      vector[4] new_log_alpha;
      for (j in 1:4) {
        vector[4] temp;
        for (i in 1:4) temp[i] = log_alpha[i] + log(fmax(P[i,j], 1e-12));
        new_log_alpha[j] = log_sum_exp(temp) + log(fmax(e[j], 1e-12));
      }
      log_alpha = new_log_alpha;
    }
    log_lik_m[m] = log_sum_exp(log_alpha);
  }
}

model {
  // priors
  baseline_rates ~ exponential(1);
  beta_form ~ normal(0,1);
  beta_lemma ~ normal(0,1);
  beta_bipartite_raw ~ normal(0,1);
  variety_eff_raw ~ normal(0,1);
  corpus_eff_raw ~ normal(0,1);
  lemma_eff_raw ~ normal(0,1);
  sigma_variety ~ exponential(1);
  sigma_corpus ~ exponential(1);
  sigma_lemma ~ exponential(1);

  for (l in 1:N_lemmas) target += dirichlet_lpdf(pi_lemma[l] | alpha_prior);

  emission_logit_base ~ normal(1.5, 1.0);
  emission_variety_raw ~ normal(0,1);
  emission_corpus_raw ~ normal(0,1);
  emission_pp_raw ~ normal(0,1);
  sigma_em_var ~ exponential(1);
  sigma_em_corp ~ exponential(1);
  sigma_em_pp ~ exponential(1);
  for (s in 1:4) emission_off[s] ~ dirichlet(rep_vector(1.0, 3));
  beta_emiss_form ~ normal(0, 0.5);
  beta_emiss_logtime ~ normal(0, 0.5);

  // sequence marginal log-likelihood = log_sum_exp(log_alpha)
  target += sum(log_lik_m);
}


generated quantities {
  vector[M] log_lik = log_lik_m;
}
