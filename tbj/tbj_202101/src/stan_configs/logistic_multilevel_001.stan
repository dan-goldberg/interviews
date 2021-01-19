// This model keeps every feature and parameter separate. I tried doing it in a more vectorized way
// but ran into serious convergence issues. So the model is more bespoke (classic Bayesian) but
// actually works on my data, which is nice!
data {
  int<lower=0> num_samples;
  int<lower=1> num_features;

  vector[num_samples] feature1;
  vector[num_samples] feature2;
  vector[num_samples] feature3;
  vector[num_samples] feature4;
  vector[num_samples] feature5;
  vector[num_samples] feature6;
  vector[num_samples] feature7;
  vector[num_samples] feature8;
  vector[num_samples] feature9;
  vector[num_samples] feature10;

  int<lower=0,upper=1> labels[num_samples];

  // variable effects
  int<lower=1> num_levels;
  int<lower=1,upper=num_levels> level[num_samples];
}
parameters {
  real slope1;
  real slope2;
  real slope3;
  real slope4;
  real slope5;
  real slope6;
  real slope7;
  real slope8;
  real slope9;
  real slope10;
  real bias;
  
  // variable effects
  real<lower=0> sigma;
  real shortstop_effect[num_levels];
}
model {
  vector[num_samples] x_beta_ll;

  slope1 ~ normal(0, 1);
  slope2 ~ normal(0, 1);
  slope3 ~ normal(0, 1);
  slope4 ~ normal(0, 1);
  slope5 ~ normal(0, 1);
  slope6 ~ normal(0, 1);
  slope7 ~ normal(0, 1);
  slope8 ~ normal(0, 1);
  slope9 ~ normal(0, 1);
  slope10 ~ normal(0, 1);
  bias ~ normal(0, 1);

  sigma ~ exponential(1);
  for (l in 1:num_levels) {
    shortstop_effect[l] ~ normal(0, sigma);
  }
  for (n in 1:num_samples) {
      x_beta_ll[n] = bias
        + feature1[n] * slope1 
        + feature2[n] * slope2 
        + feature3[n] * slope3 
        + feature4[n] * slope4 
        + feature5[n] * slope5 
        + feature6[n] * slope6 
        + feature7[n] * slope7 
        + feature8[n] * slope8 
        + feature9[n] * slope9 
        + feature10[n] * slope10 
        + shortstop_effect[level[n]];
  }
  labels ~ bernoulli_logit(x_beta_ll);
}