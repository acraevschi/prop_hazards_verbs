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
      real lr = log(base + 1e-9) + beta_form * form_freq + beta_lemma * lemma_freq + beta_bipartite[i] + variety_eff + corpus_eff + lemma_eff;
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
  int<lower=1> M; // number of sequences
  int<lower=1> max_T;
  int<lower=1> T_len[M];
  int<lower=1,upper=4> obs_states[M, max_T];
  real time_intervals[M, max_T-1];

  int<lower=1> N_lemmas;
  int<lower=1> N_varieties;
  int<lower=1> N_corpora;
  real form_freq[M, max_T];
  real lemma_freq[M, max_T];
  int<lower=1,upper=N_lemmas> lemma_id[M];
  int<lower=1,upper=4> principal_part[M];
  int<lower=1,upper=N_varieties> variety[M];
  int<lower=1,upper=N_corpora> corpus[M];
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
  real<lower=0, upper=0.25> measurement_error;
}

transformed parameters {
  vector[N_varieties] variety_eff = sigma_variety * variety_eff_raw;
  vector[N_corpora] corpus_eff = sigma_corpus * corpus_eff_raw;
  vector[N_lemmas] lemma_eff = sigma_lemma * lemma_eff_raw;
  vector[4] beta_bipartite;
  beta_bipartite[1:3] = beta_bipartite_raw;
  beta_bipartite[4] = 0;
}

model {
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
  measurement_error ~ beta(10, 90);

  // Sequential likelihood across each sequence: product of transition probabilities for observed pairs
  for (m in 1:M) {
    for (t in 1:(T_len[m]-1)) {
      int s = obs_states[m,t];
      int e = obs_states[m,t+1];
      real interval = time_intervals[m,t];
      
      if (interval <= 0 && s != e) {
        // Zero-interval change: treat as potential measurement error
        target += log(measurement_error) - log(3);  // uniform over other states
      } else if (interval <= 0 && s == e) {
        // Zero-interval, same state: high confidence
        target += log1p(-measurement_error);
      } else {
        // Positive interval: use continuous-time model
        matrix[4,4] P = get_transition_matrix(baseline_rates, beta_form, beta_lemma, beta_bipartite,
                                            variety_eff[variety[m]], corpus_eff[corpus[m]], lemma_eff[lemma_id[m]],
                                            form_freq[m,t], lemma_freq[m,t], (s==4) ? 1 : 0, time_intervals[m,t]);
        target += log(P[s,e] + 1e-10);
      }
    }
  }
}

// generated quantities {
//   vector[M] log_lik_seq;
//   for (m in 1:M) {
//     real lp = 0;
//     for (t in 1:(T_len[m]-1)) {
//       int s = obs_states[m,t];
//       int e = obs_states[m,t+1];
//       real ft = form_freq[m,t];
//       real lt = lemma_freq[m,t];
//       matrix[4,4] P = get_transition_matrix(baseline_rates, beta_form, beta_lemma, beta_bipartite,
//                                             sigma_variety*variety_eff_raw[variety[m]], sigma_corpus*corpus_eff_raw[corpus[m]], sigma_lemma*lemma_eff_raw[lemma_id[m]],
//                                             ft, lt, (s==4) ? 1 : 0, time_intervals[m,t]);
//       lp += log(P[s,e]);
//     }
//     log_lik_seq[m] = lp;
//   }
// }
