data {
  int<lower=1> N_verbs;
  int<lower=1> N_obs;
  int<lower=1> N_states;           // still 4
  int<lower=1> N_chains;           // = 3 sub‑paradigms
  int<lower=1> form[N_obs];
  real time_since_prev[N_obs];
  int<lower=0,upper=1> obs_v[N_obs], obs_c[N_obs];
  real freq[N_obs];
  int n_dialects;
  int<lower=1> dialect_id[N_obs];
  int<lower=1> verb_starts[N_verbs], verb_ends[N_verbs];
  int<lower=1> num_basis;
  matrix[N_obs,num_basis] basis;
  // NEW: which chain (1–3) each observation belongs to
  int<lower=1,upper=N_chains> chain_id[N_obs];
}

parameters {
  // base rates & freq effects for each chain
  matrix[N_states,N_states] log_lambda[N_chains];
  matrix[N_states,N_states] beta_trans[N_chains];

  // COUPLING parameters: how chain g′ in state i affects chain g’s rate out of i→j
  // For simplicity a single shared coupling weight per pair (g, g′):
  real gamma[N_chains, N_chains];

  // emissions etc.
  vector[n_dialects] beta_dialects;
  simplex[N_states] initial_probs[N_chains];
  vector[num_basis] beta_v_true1, beta_v_true0, beta_c_true1, beta_c_true0;
}

model {
  // priors… (as before, for each chain separately)
  for (g in 1:N_chains) {
    to_vector(log_lambda[g]) ~ normal(0,1);
    to_vector(beta_trans[g]) ~ normal(0,1);
    initial_probs[g] ~ dirichlet(rep_vector(1.0, N_states)); // this will be modified later, make prior more likely to be the same as observed counts at t=0
  }
  gamma ~ normal(0, 1);

  // loop over verbs
  for (v in 1:N_verbs) {
    int T = verb_ends[v] - verb_starts[v] + 1;
    int idxs[T];
    for (t in 1:T) idxs[t] = verb_starts[v] + t - 1;

    // we now need a **joint** forward pass across ALL CHAINS
    // but to keep dimensionality manageable, we do a **mean-field** approximation:
    // at each time t we track, for each chain g, a vector log_forward[g][t][s]
    // and we use the *expected* states of other chains under the previous forward
    // distribution to build each chain’s Q.

    vector[N_states] log_forward[N_chains, T];

    // initialize each chain g
    for (g in 1:N_chains) {
      log_forward[g,1] = log(initial_probs[g]);
      // add emission of the very first obs in chain g (if it appears at t=1)
      if (chain_id[idxs[1]] == g) {
        int i = idxs[1];
        for (s in 1:N_states) {
            // Emission logic for each state s and observation i
            int v_true;
            int c_true;

            if (s <= 2) {
                v_true = 1;
            } else {
                v_true = 0;
            }

            if (s % 2 == 1) {
                c_true = 1;
            } else {
                c_true = 0;
            }

            real logit_v;
            if (v_true == 1) {
                logit_v = dot_product(basis[i], beta_v_true1) + beta_dialects[dialect_id[i]];
            } else {
                logit_v = dot_product(basis[i], beta_v_true0) + beta_dialects[dialect_id[i]];
            }

            real logit_c;
            if (c_true == 1) {
                logit_c = dot_product(basis[i], beta_c_true1) + beta_dialects[dialect_id[i]];
            } else {
                logit_c = dot_product(basis[i], beta_c_true0) + beta_dialects[dialect_id[i]];
            }
            
          log_forward[g,1][s] += bernoulli_logit_lpmf(obs_v[i] | logit_v)
                              + bernoulli_logit_lpmf(obs_c[i] | logit_c);
        }
      }
    }

    // subsequent times
    for (t in 2:T) {
      // first compute the *marginal* posterior over states at t−1 for each chain
      // (i.e. normalized exp(log_forward[g,t−1]))
      vector[N_states] post_prev[N_chains];
      for (g in 1:N_chains) {
        post_prev[g] = softmax(log_forward[g,t-1]);
      }

      int i_prev = idxs[t-1];
      int i_curr = idxs[t];
      real dt = time_since_prev[i_curr];

      // update each chain g **in parallel**, using the *expected* influence
      for (g in 1:N_chains) {
        // build its rate matrix Q_g
        matrix[N_states,N_states] Q_g = rep_matrix(0.0, N_states, N_states);

        for (i_s in 1:N_states) {
          real total_rate = 0;
          for (j_s in 1:N_states) if (i_s != j_s) {
            // base rate + frequency effect
            real logr = log_lambda[g][i_s, j_s] 
                      + beta_trans[g][i_s, j_s] * freq[i_prev];
            // PLUS coupling: add gamma[g, g′] times probability that chain g′ was in state i_s
            real coup = 0;
            for (g2 in 1:N_chains) if (g2 != g)
              coup += gamma[g, g2] * post_prev[g2][i_s];
            logr += coup;

            Q_g[i_s, j_s] = exp(logr);
            total_rate += Q_g[i_s, j_s];
          }
          Q_g[i_s, i_s] = -total_rate;
        }

        // compute P_g = expm(Q_g * dt), then the log-transition matrix
        matrix[N_states,N_states] P_g = matrix_exp(Q_g * dt);
        matrix[N_states,N_states] logP_g = log(P_g + 1e-9);

        // transition step for forward
        vector[N_states] tmp;
        for (j_s in 1:N_states)
          tmp[j_s] = log_sum_exp(log_forward[g,t-1] + logP_g[, j_s]);
        log_forward[g,t] = tmp;

        // emission for chain g if obs belongs here
        if (chain_id[i_curr] == g) {
          for (s in 1:N_states) {
            // Emission logic for each state s and observation i
            int v_true;
            int c_true;

            if (s <= 2) {
                v_true = 1;
            } else {
                v_true = 0;
            }

            if (s % 2 == 1) {
                c_true = 1;
            } else {
                c_true = 0;
            }

            real logit_v;
            if (v_true == 1) {
                logit_v = dot_product(basis[i], beta_v_true1) + beta_dialects[dialect_id[i]];
            } else {
                logit_v = dot_product(basis[i], beta_v_true0) + beta_dialects[dialect_id[i]];
            }

            real logit_c;
            if (c_true == 1) {
                logit_c = dot_product(basis[i], beta_c_true1) + beta_dialects[dialect_id[i]];
            } else {
                logit_c = dot_product(basis[i], beta_c_true0) + beta_dialects[dialect_id[i]];
            }
            
            log_forward[g,t][s] += bernoulli_logit_lpmf(obs_v[i_curr] | logit_v)
                                + bernoulli_logit_lpmf(obs_c[i_curr] | logit_c);
            }
        }
      }
    } // end t

    // finally, add log-likelihoods of each chain’s final forward
    for (g in 1:N_chains)
      target += log_sum_exp(log_forward[g, T]);
  } // end v
}
