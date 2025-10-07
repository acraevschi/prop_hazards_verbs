functions {
  matrix get_transition_matrix(vector rates, real beta_form, real beta_lemma,
                               vector beta_bipartite, real variety_eff,
                               real corpus_eff, real lemma_eff,
                               real form_freq, real lemma_freq,
                               int is_bipartite, real time_interval) {
    matrix[4,4] Q = rep_matrix(0, 4, 4);
    int idx = 1;
    
    for (i in 1:4) for (j in 1:4) if (i != j) {
      real lr = log(rates[idx]) + beta_form * form_freq + beta_lemma * lemma_freq + 
                beta_bipartite[i] + variety_eff + corpus_eff + lemma_eff;
      Q[i,j] = exp(lr);
      idx += 1;
    }
    
    for (i in 1:4) Q[i,i] = -sum(Q[i,]);
    
    return (time_interval <= 0) ? diag_matrix(rep_vector(1.0, 4)) : matrix_exp(Q * time_interval);
  }
  
  vector get_emission_probs(int obs_state, vector emission_logit_base,
                           real variety_eff, real corpus_eff, real pp_eff,
                           real beta_emiss_form, real beta_emiss_logtime,
                           real form_freq, real log_time, matrix emission_off) {
    vector[4] diag_logit = emission_logit_base + variety_eff + corpus_eff + pp_eff + 
                           beta_emiss_form * form_freq + beta_emiss_logtime * log_time;
    vector[4] diag_p = inv_logit(diag_logit);
    vector[4] emission_probs;
    
    for (z in 1:4) {
      if (obs_state == z) {
        emission_probs[z] = diag_p[z];
      } else {
        // Find the position in off-diagonal weights
        int k = (obs_state < z) ? obs_state : obs_state - 1;
        emission_probs[z] = (1 - diag_p[z]) * emission_off[z, k];
      }
    }
    
    return emission_probs;
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
  vector[4] alpha_prior;
}

transformed data {
  // Precompute log(time + 1) for all observations
  real log_time_intervals[M, max_T-1];
  for (m in 1:M) {
    for (t in 1:(T_len[m]-1)) {
      log_time_intervals[m, t] = log1p(time_intervals[m, t]);
    }
  }
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

  simplex[4] pi_lemma[N_lemmas];

  // Emission parameters - consolidated
  vector[4] emission_logit_base;
  vector[N_varieties] emission_variety_raw;
  vector[N_corpora] emission_corpus_raw;
  vector[4] emission_pp_raw;
  real<lower=0> sigma_em_var;
  real<lower=0> sigma_em_corp;
  real<lower=0> sigma_em_pp;
  
  // Emission off-diagonals as matrix for efficiency
  simplex[3] emission_off_simplex[4];
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
  
  // Convert emission_off to matrix for efficient access
  matrix[4, 3] emission_off;
  for (s in 1:4) {
    emission_off[s] = to_row_vector(emission_off_simplex[s]);
  }
  
  vector[M] log_lik_m;

  // Main forward algorithm - optimized
  for (m in 1:M) {
    int Tm = T_len[m];
    vector[4] log_alpha;
    int lem_id = lemma_id[m];
    int var_id = variety[m];
    int corp_id = corpus[m];
    int pp_id = principal_part[m];
    
    real var_eff_em = emission_variety[var_id];
    real corp_eff_em = emission_corpus[corp_id];
    real pp_eff_em = emission_pp[pp_id];
    
    real var_eff_trans = variety_eff[var_id];
    real corp_eff_trans = corpus_eff[corp_id];
    real lem_eff_trans = lemma_eff[lem_id];

    // t = 1
    vector[4] e1 = get_emission_probs(obs_states[m, 1], emission_logit_base,
                                     var_eff_em, corp_eff_em, pp_eff_em,
                                     beta_emiss_form, beta_emiss_logtime,
                                     form_freq[m, 1], 
                                     (Tm > 1) ? log_time_intervals[m, 1] : 0,
                                     emission_off);
    
    for (z in 1:4) {
      log_alpha[z] = log(pi_lemma[lem_id][z]) + log(e1[z]);
    }

    // t = 2..Tm
    for (t in 2:Tm) {
      matrix[4,4] P = get_transition_matrix(baseline_rates, beta_form, beta_lemma, beta_bipartite,
                                           var_eff_trans, corp_eff_trans, lem_eff_trans,
                                           form_freq[m, t-1], lemma_freq[m, t-1], 
                                           (obs_states[m, t-1] == 4) ? 1 : 0, 
                                           time_intervals[m, t-1]);
      
      vector[4] e = get_emission_probs(obs_states[m, t], emission_logit_base,
                                      var_eff_em, corp_eff_em, pp_eff_em,
                                      beta_emiss_form, beta_emiss_logtime,
                                      form_freq[m, t], log_time_intervals[m, t-1],
                                      emission_off);
      
      vector[4] new_log_alpha;
      for (j in 1:4) {
        new_log_alpha[j] = log_sum_exp(log_alpha + log(P[, j])) + log(e[j]);
      }
      log_alpha = new_log_alpha;
    }
    
    log_lik_m[m] = log_sum_exp(log_alpha);
  }
}

model {
  // priors (unchanged)
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

  for (l in 1:N_lemmas) {
    pi_lemma[l] ~ dirichlet(alpha_prior);
  }

  emission_logit_base ~ normal(1.5, 1.0);
  emission_variety_raw ~ normal(0,1);
  emission_corpus_raw ~ normal(0,1);
  emission_pp_raw ~ normal(0,1);
  sigma_em_var ~ exponential(1);
  sigma_em_corp ~ exponential(1);
  sigma_em_pp ~ exponential(1);
  
  for (s in 1:4) {
    emission_off_simplex[s] ~ dirichlet(rep_vector(1.0, 3));
  }
  
  beta_emiss_form ~ normal(0, 0.5);
  beta_emiss_logtime ~ normal(0, 0.5);

  target += sum(log_lik_m);
}

generated quantities {
  vector[M] log_lik = log_lik_m;
}