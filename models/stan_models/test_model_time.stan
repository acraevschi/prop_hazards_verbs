data {
  int<lower=1> N_verbs;                // Number of verbs
  int<lower=1> N_obs;                  // Total observations across all verbs
  int<lower=1> N_states;               // 4 states: (1,1), (1,0), (0,1), (0,0)
  int<lower=1> form[N_obs];            // Form ID for each observation
  real time[N_obs];                    // Time of observation
  real time_since_prev[N_obs];         // Time since last observation (0 for first)
  int<lower=0, upper=1> obs_v[N_obs];  // Observed vowel alternation (0/1)
  int<lower=0, upper=1> obs_c[N_obs];  // Observed consonant alternation (0/1)
  real freq[N_obs];                    // Frequency covariate
  int n_dialects;                      // Number of dialects
  int<lower=1> dialect_id[N_obs];      // Dialect ID for each observation, covariate for emission probabilities
  int<lower=1> verb_starts[N_verbs];   // Starting index for each form
  int<lower=1> verb_ends[N_verbs];     // Ending index for each form
  int<lower=1> num_basis;              // Number of spline basis functions
  matrix[N_obs, num_basis] basis;      // B-spline basis matrix for all observations
}

parameters {
  // Transition parameters (baseline rates and frequency effects)
  matrix[N_states, N_states] log_lambda;             // Baseline log transition rates (diag ignored)
  matrix[N_states, N_states] beta_trans;             // Frequency effect on transition rates

  // add parameter for each dialect group
  vector[n_dialects] beta_dialects; // Dialect effects on emission probabilities

  simplex[N_states] initial_probs; // Initial state probabilities

  // Spline coefficients for emission probabilities | maybe not necessary to distinguish between vowels and consonants
  vector[num_basis] beta_v_true1;      // Vowel true=1 coefficients
  vector[num_basis] beta_v_true0;      // Vowel true=0 coefficients
  vector[num_basis] beta_c_true1;      // Consonant true=1 coefficients
  vector[num_basis] beta_c_true0;      // Consonant true=0 coefficients
}

model {
  // Priors
  to_vector(log_lambda) ~ normal(0, 1);
  to_vector(beta_trans) ~ normal(0, 1);
  beta_v_true1 ~ normal(0, 1);
  beta_v_true0 ~ normal(0, 1);
  beta_c_true1 ~ normal(0, 1);
  beta_c_true0 ~ normal(0, 1);
  beta_dialects ~ normal(0, 1);
  // Prior for initial state probabilities; 
  initial_probs ~ dirichlet([3.0, 2.0, 2.0, 1.0]');
  // TODO: Flat with values of 1, and add 1 for each count of the state in the data at time t=0
  // This will bias the initial state probabilities towards the observed counts in the data
  

  for (v in 1:N_verbs) {
    // Dealing with observations for each verb
    // Get the start and end indices for the current verb
    int T = verb_ends[v] - verb_starts[v] + 1;
    int obs_v_indices[T];
    for (i in 1:T) {
      obs_v_indices[i] = verb_starts[v] + i - 1;
    }

    vector[N_states] log_forward[T];

    // Initialize with the prior
    log_forward[1] = log(initial_probs);

    // First observation
    for (s in 1:N_states) {
      int v_true;
      int c_true;
  
      if (s <= 2) {  // States 1-2 have vowel alternation
        v_true = 1;
      } else {        // States 3-4 don't
        v_true = 0;
      }
  
      if (s % 2 == 1) {  // Odd states (1,3) have consonant alternation
        c_true = 1;
      } else {            // Even states (2,4) don't
        c_true = 0;
      }

      int idx = obs_v_indices[1];

      // Compute emission logits using splines
      real logit_v;
      if (v_true == 1) {
        logit_v = dot_product(basis[idx, :], beta_v_true1) + beta_dialects[dialect_id[idx]];
      } else {
        logit_v = dot_product(basis[idx, :], beta_v_true0) + beta_dialects[dialect_id[idx]];
      }

      real logit_c; 
      if (c_true == 1) {
        logit_c = dot_product(basis[idx, :], beta_c_true1) + beta_dialects[dialect_id[idx]];
      } else {
        logit_c = dot_product(basis[idx, :], beta_c_true0) + beta_dialects[dialect_id[idx]];
      }

      log_forward[1][s] += bernoulli_logit_lpmf(obs_v[idx] | logit_v) + bernoulli_logit_lpmf(obs_c[idx] | logit_c);
    }

    // Subsequent observations
    for (t in 2:T) {
      real delta_t = time_since_prev[obs_v_indices[t]];
      int prev_idx = obs_v_indices[t-1];
      int curr_idx = obs_v_indices[t];

      // Construct rate matrix with frequency effect
      matrix[N_states, N_states] Q = rep_matrix(0.0, N_states, N_states);
      for (i in 1:N_states) {
        real total_rate = 0.0;
        for (j in 1:N_states) {
            if (i != j) {
                // Get transition rate
                Q[i, j] = exp(log_lambda[i, j] + beta_trans[i, j] * freq[prev_idx]) + 1e-9; // Avoid zero rates
                total_rate += Q[i, j];
            }
        }
            Q[i, i] = -total_rate;
            beta_trans[i, i] = 0; // Set diagonal to zero
      }

      matrix[N_states, N_states] P = matrix_exp(Q * delta_t);
      matrix[N_states, N_states] log_P = log(P + 1e-9); // Avoid log(0)

      // Update forward probabilities
      for (j in 1:N_states) {
        log_forward[t][j] = log_sum_exp(log_forward[t-1] + log_P[, j]);
      }

      // Apply emission probabilities
      for (s in 1:N_states) {

        int v_true;
        int c_true;
    
        if (s <= 2) {  // States 1-2 have vowel alternation
            v_true = 1;
        } else {        // States 3-4 don't
            v_true = 0;
        }

        if (s % 2 == 1) {  // Odd states (1,3) have consonant alternation
            c_true = 1;
        } else {            // Even states (2,4) don't
            c_true = 0;
        }

        real logit_v;
        if (v_true == 1) {
            logit_v = dot_product(basis[curr_idx, :], beta_v_true1) + beta_dialects[dialect_id[curr_idx]];
        } else {
            logit_v = dot_product(basis[curr_idx, :], beta_v_true0) + beta_dialects[dialect_id[curr_idx]];
        }

        real logit_c; 
        if (c_true == 1) {
            logit_c = dot_product(basis[curr_idx, :], beta_c_true1) + beta_dialects[dialect_id[curr_idx]];
        } else {
            logit_c = dot_product(basis[curr_idx, :], beta_c_true0) + beta_dialects[dialect_id[curr_idx]];
        }

        log_forward[t][s] += bernoulli_logit_lpmf(obs_v[curr_idx] | logit_v) + bernoulli_logit_lpmf(obs_c[curr_idx] | logit_c);
      }
    }

    target += log_sum_exp(log_forward[T]);
  }
}
