data {
  // Original variables
  int<lower=1> N_verbs;                // Number of verb forms
  int<lower=1> N_obs;                  // Total observations across all verbs
  int<lower=1> N_states;               // 4 states: (1,1), (1,0), (0,1), (0,0)
  int<lower=1> form[N_obs];            // Form ID for each observation
  real time[N_obs];                    // Time of observation
  real time_since_prev[N_obs];         // Time since last observation (0 for first)
  int<lower=0, upper=1> obs_v[N_obs];  // Observed vowel alternation (0/1)
  int<lower=0, upper=1> obs_c[N_obs];  // Observed consonant alternation (0/1)
  real freq[N_obs];                    // Frequency covariate
  int n_dialects;                      // Number of dialects
  int<lower=1> dialect_id[N_obs];      // Dialect ID for each observation
  int<lower=1> verb_starts[N_verbs];   // Starting index for each form
  int<lower=1> verb_ends[N_verbs];     // Ending index for each form
  int<lower=1> num_basis;              // Number of spline basis functions
  matrix[N_obs, num_basis] basis;      // B-spline basis matrix for observations
  
  // New variables for paradigm model
  int<lower=1> N_lemmas;               // Number of unique lemmas
  int<lower=1> lemma_id[N_obs];        // Which lemma each observation belongs to
  int<lower=1> N_principal_parts;      // Number of principal parts (e.g., 3 for infinitive/past/participle)
  int<lower=1> principal_part_id[N_obs]; // Which principal part each observation represents
  
  // Time-varying irregularity index
  int<lower=1> n_time_points;          // Number of unique time points in data
  real unique_times[n_time_points];    // Array of unique time points
  matrix[N_lemmas, n_time_points] irregularity_index; // Pre-computed irregularity at each time
  matrix[N_lemmas, n_time_points] m_values; // Number of principal parts observed at each time
}

parameters {
  // Original parameters
  matrix[N_states, N_states] log_lambda;    // Baseline log transition rates (diag ignored)
  matrix[N_states, N_states] beta_trans;    // Frequency effect on transition rates
  vector[n_dialects] beta_dialects;         // Dialect effects on emission probabilities
  simplex[N_states] initial_probs;          // Initial state probabilities
  
  // Spline coefficients for emission probabilities
  vector[num_basis] beta_v_true1;           // Vowel true=1 coefficients
  vector[num_basis] beta_v_true0;           // Vowel true=0 coefficients
  vector[num_basis] beta_c_true1;           // Consonant true=1 coefficients
  vector[num_basis] beta_c_true0;           // Consonant true=0 coefficients
  
  // New parameters for paradigm effects
  real gamma_irreg;                         // Effect of irregularity on paradigm leveling

  matrix[N_principal_parts, N_principal_parts] paradigm_influence_raw; // Cross-influence between parts
  // Hierarchical structure for lemma-specific irregularity adjustment
  real<lower=0> irregularity_adjustment_sd;    // Population-level standard deviation
  vector<lower=0>[N_lemmas] irregularity_adjustment;  // Lemma-specific adjustments
  // note 0 boundary because irregularity cannot decrease after observing more forms
}

transformed parameters {
  matrix[N_lemmas, n_time_points] true_irregularity_index;
  // New parameter for paradigm influence with fixed diagonal
  matrix[N_principal_parts, N_principal_parts] paradigm_influence;
  
  // Create paradigm_influence with fixed diagonal elements
  for (i in 1:N_principal_parts) {
    for (j in 1:N_principal_parts) {
      if (i == j) {
        paradigm_influence[i, j] = 0.0;  // Fix diagonal to 0.0 (or any other constant)
      } else {
        paradigm_influence[i, j] = paradigm_influence_raw[i, j];
      }
    }
  }

  matrix[N_lemmas, n_time_points] true_irregularity_index;
  
  for (l in 1:N_lemmas) {
    for (t in 1:n_time_points) {
      // More missing forms = more potential hidden irregularity
      real missing_forms_proportion = (N_principal_parts - m_values[l,t]) / 
                                      (N_principal_parts * 1.0);
      
      // Use lemma-specific adjustment (l) but apply to all time points
      true_irregularity_index[l,t] = inv_logit(irregularity_index[l,t] + 
                                     irregularity_adjustment[l] * missing_forms_proportion);
    }
  }
}

model {
  // Original priors
  to_vector(log_lambda) ~ normal(0, 1);
  to_vector(beta_trans) ~ normal(0, 1);
  beta_v_true1 ~ normal(0, 1);
  beta_v_true0 ~ normal(0, 1);
  beta_c_true1 ~ normal(0, 1);
  beta_c_true0 ~ normal(0, 1);
  beta_dialects ~ normal(0, 1);
  initial_probs ~ dirichlet([3.0, 2.0, 2.0, 1.0]');
  
  // New priors for paradigm effects
  gamma_irreg ~ normal(0, 1);  // Can be positive (supports Paul) or negative (rejects Paul)
  to_vector(paradigm_influence_raw) ~ normal(0.5, 0.5); // Cross-influence between parts
  
  // Hyper-priors for irregularity adjustment
  irregularity_adjustment_sd ~ exponential(2);
  
  // Prior for lemma-specific adjustments
  irregularity_adjustment ~ normal(0, irregularity_adjustment_sd);

  // Main model logic
  for (v in 1:N_verbs) {
    // Dealing with observations for each verb
    int T = verb_ends[v] - verb_starts[v] + 1;
    int obs_verb_indices[T];
    for (i in 1:T) {
      obs_verb_indices[i] = verb_starts[v] + i - 1;
    }

    vector[N_states] log_forward[T];

    // Initialize with the prior
    log_forward[1] = log(initial_probs);

    // First observation - emission probabilities
    for (s in 1:N_states) {
      int v_true;
      int c_true;

      // Replace ternary operator with if-else
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

      int idx = obs_verb_indices[1];

      // Compute emission logits using splines
      real logit_v;
      if (v_true == 1) {
        logit_v = dot_product(basis[idx, :], beta_v_true1);
      } else {
        logit_v = dot_product(basis[idx, :], beta_v_true0);
      }
      logit_v += beta_dialects[dialect_id[idx]];

      real logit_c;
      if (c_true == 1) {
        logit_c = dot_product(basis[idx, :], beta_c_true1);
      } else {
        logit_c = dot_product(basis[idx, :], beta_c_true0);
      }
      logit_c += beta_dialects[dialect_id[idx]];

      log_forward[1][s] += bernoulli_logit_lpmf(obs_v[idx] | logit_v) + 
                           bernoulli_logit_lpmf(obs_c[idx] | logit_c);
    }


    // Subsequent observations
    for (t in 2:T) {
      real delta_t = time_since_prev[obs_verb_indices[t]];
      int prev_idx = obs_verb_indices[t-1];
      int curr_idx = obs_verb_indices[t];
      int curr_lemma = lemma_id[curr_idx];
      int curr_pp = principal_part_id[curr_idx];

      // Need to find the correct time point index
      int time_point_idx = 1;  // Default to first time point
      real curr_time = time[curr_idx];

      // Find the appropriate time point (binary search would be better for large datasets)
      for (tp in 1:n_time_points) {
        if (abs(unique_times[tp] - curr_time) < 1e-6) {  // checks if it's matching time point
            time_point_idx = tp;
            break;
        }
      }
      
      // Track influence from other principal parts
      vector[N_states] state_influence = rep_vector(0.0, N_states);
      
      // For each principal part of the same lemma
      for (other_pp in 1:N_principal_parts) {
        if (other_pp != curr_pp) {
          // Find most recent observation of this principal part for this lemma
          int other_latest_idx = 0;
          
          // Find the most recent observation for this lemma+principal part
          for (j in 1:N_obs) {
            if (lemma_id[j] == curr_lemma && 
                principal_part_id[j] == other_pp && 
                time[j] < time[curr_idx] &&
                (other_latest_idx == 0 || time[j] > time[other_latest_idx])) {
              other_latest_idx = j;
            }
          }
          
          if (other_latest_idx > 0) {
            // We found an observation for this principal part
            // Simple approximation: use observed vowel/consonant patterns to infer state
            int v_obs = obs_v[other_latest_idx];
            int c_obs = obs_c[other_latest_idx];
            
            // Create a rough estimate of state probabilities from observations
            vector[N_states] other_state_probs;
            for (s in 1:N_states) {
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

              
              // Calculate emission probabilities using the model's parameters
              real logit_v_other;
              if (v_true == 1) {
                logit_v_other = dot_product(basis[other_latest_idx, :], beta_v_true1);
              } else {
                logit_v_other = dot_product(basis[other_latest_idx, :], beta_v_true0);
              }
              logit_v_other += beta_dialects[dialect_id[other_latest_idx]];

              real logit_c_other;
              if (c_true == 1) {
                logit_c_other = dot_product(basis[other_latest_idx, :], beta_c_true1);
              } else {
                logit_c_other = dot_product(basis[other_latest_idx, :], beta_c_true0);
              }
              logit_c_other += beta_dialects[dialect_id[other_latest_idx]];

              // Convert to probabilities
              real p_v = inv_logit(logit_v_other);
              real p_c = inv_logit(logit_c_other);

              // Calculate joint probability (includes both match and mismatch cases)
              if (v_obs == 1) {
                if (c_obs == 1) {
                  other_state_probs[s] = p_v * p_c;
                } else {
                  other_state_probs[s] = p_v * (1 - p_c);
                }
              } else {
                if (c_obs == 1) {
                  other_state_probs[s] = (1 - p_v) * p_c;
                } else {
                  other_state_probs[s] = (1 - p_v) * (1 - p_c);
                }
              }
            }
            
            // Normalize probabilities
            other_state_probs = other_state_probs / sum(other_state_probs);
            
            // Apply the paradigm influence
            for (s in 1:N_states) {
              state_influence[s] += paradigm_influence[curr_pp, other_pp] * other_state_probs[s];
            }
          }
        }
      }

      // Construct rate matrix with frequency and paradigm effects
      matrix[N_states, N_states] Q = rep_matrix(0.0, N_states, N_states);
      for (i in 1:N_states) {
        real total_rate = 0.0;
        for (j in 1:N_states) {
          if (i != j) {
            // Is this a regularizing transition?
            int is_regularizing;
            if ((i <= 2 && j > 2) || (i % 2 == 1 && j % 2 == 0)) {
              is_regularizing = 1;
            } else {
              is_regularizing = 0;
            }
            
            // The key parameter to test Paul's hypothesis
            real irregularity_effect = gamma_irreg * true_irregularity_index[curr_lemma, time_point_idx] * is_regularizing;
            
            // Get transition rate with all effects
            Q[i, j] = exp(
              log_lambda[i, j] + 
              beta_trans[i, j] * freq[prev_idx] -
              irregularity_effect +
              state_influence[j]  // Influence toward state j
            ) + 1e-9;  // Avoid zero rates
            
            total_rate += Q[i, j];
          }
        }
        Q[i, i] = -total_rate;
      }

      // Calculate transition probability matrix
      matrix[N_states, N_states] P = matrix_exp(Q * delta_t);
      matrix[N_states, N_states] log_P = log(P + 1e-9); // Avoid log(0)

      // Forward algorithm update
      for (j in 1:N_states) {
        log_forward[t][j] = log_sum_exp(log_forward[t-1] + log_P[, j]);
      }

      // Apply emission probabilities
      for (s in 1:N_states) {
        int v_true;
        int c_true;

        // Replace ternary operator with if-else
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
          logit_v = dot_product(basis[curr_idx, :], beta_v_true1);
        } else {
          logit_v = dot_product(basis[curr_idx, :], beta_v_true0);
        }
        logit_v += beta_dialects[dialect_id[curr_idx]];

        real logit_c;
        if (c_true == 1) {
          logit_c = dot_product(basis[curr_idx, :], beta_c_true1);
        } else {
          logit_c = dot_product(basis[curr_idx, :], beta_c_true0);
        }
        logit_c += beta_dialects[dialect_id[curr_idx]];

        log_forward[t][s] += bernoulli_logit_lpmf(obs_v[curr_idx] | logit_v) + 
                            bernoulli_logit_lpmf(obs_c[curr_idx] | logit_c);
      }
    }
    
    // Add log likelihood for this verb
    target += log_sum_exp(log_forward[T]);
  }
}

generated quantities {
  // Generate interpretable quantities
  real paradigm_effect = gamma_irreg;  // Direct test of Hermann Paul's hypothesis
}
