
// data {
//     // neural network input
//     int<lower=0> num_samples; // number of samples
//     int<lower=1> num_features; // number of sample features (excluding cluster ids)
//     row_vector[num_features] feature_vectors[num_samples]; // data matrix

//     // // multilevel clusters
//     // int<lower=1> num_shortstops; // number of shortstops
//     // int<lower=1, upper=num_shortstops> shortstop_id[num_samples]; // the id of the shortstop (re-indexed to 1) for each sample
    
//     // output
//     int<lower=0,upper=1> labels[num_samples]; // label 1 if ss gets out, 0 if not
// }
// parameters {

//     // fixed effects params
//     vector[num_features] fixed_effects; // a parameter for each feature

//     // // variable effects params
//     // real<lower=0> shortstop_effect_var;
//     // real shortstop_effect[num_shortstops];
// }
// model {
//     vector[num_samples] output;

//     // fixed effects priors
//     fixed_effects ~ normal(0.0, 0.5);

//     // // variable effect hyperpriors
//     // shortstop_effect_var ~ uniform(0.1, 10);
//     // // shortstop effect prior
//     // shortstop_effect ~ normal(0, shortstop_effect_var);

//     for (i in 1:num_samples) {
//         for (d in 1:num_features) {
//         // output[i] = shortstop_effect[shortstop_id[i]] + feature_vectors[i] * fixed_effects;
//             output[i] = feature_vectors[i], fixed_effects);
//         }
//     }
    
//     // output 
//     labels ~ bernoulli_logit(output);
// }


// data {
//   int<lower=1> num_features;
//   int<lower=0> num_samples;
//   int<lower=0,upper=1> labels[N];
  
//   row_vector[D] feature_vectors[num_samples];

//   int<lower=1> num_levels;
//   int<lower=1,upper=num_levels> level[num_samples];
// }
// parameters {
//   real mu[D];
//   real<lower=0> sigma[D];
//   vector[D] beta[L];
// }
// model {
//   for (d in 1:D) {
//     mu[d] ~ normal(0, 100);
//     for (l in 1:L)
//       beta[l,d] ~ normal(mu[d], sigma[d]);
//   }
//   for (n in 1:N)
//     y[n] ~ bernoulli(inv_logit(x[n] * beta[ll[n]]));
// }

data {
  int<lower=0> num_samples;
  int<lower=1> num_features;
  row_vector[num_features] feature_vectors[num_samples];

  int<lower=0,upper=1> labels[num_samples];

  // variable effects
  int<lower=1> num_levels;
  int<lower=1,upper=num_levels> level[num_samples];
}
parameters {
  vector[num_features] fixed_effects;
  
  // variable effects
  real<lower=0> sigma;
  real shortstop_effect[num_levels];
}
model {
  vector[num_samples] x_beta_ll;
  for (l in 1:num_levels) {
    shortstop_effect[l] ~ normal(0, sigma);
  }
  for (n in 1:num_samples) {
      x_beta_ll[n] = feature_vectors[n] * fixed_effects + shortstop_effect[level[n]];
  }
  labels ~ bernoulli_logit(x_beta_ll);
}