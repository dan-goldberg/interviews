
data {
    // neural network input
    int<lower=1> num_samples; // number of samples
    int<lower=1> num_features; // number of sample features (excluding cluster ids)
    matrix[num_samples, num_features] feature_vectors; // data matrix

    // multilevel clusters
    int<lower=1> num_shortstops; // number of shortstops
    int<lower=1, upper=num_shortstops> shortstop_id[num_samples]; // the id of the shortstop (re-indexed to 1) for each sample
    
    // output
    int<lower=0,upper=1> labels[num_samples]; // label 1 if ss gets out, 0 if not
}
parameters {

    // fixed effects params
    vector[num_features] fixed_effects; // a parameter for each feature

    // variable effects params
    real shortstop_effect_mean;
    real<lower=0> shortstop_effect_var;

    real shortstop_effect[num_shortstops];
}
transformed parameters {
    vector[num_samples] output;
    
    for (i in 1:num_samples) {
        output[i] = shortstop_effect[shortstop_id[i]] +  feature_vectors[i, ] * fixed_effects;
    }
}
model {
    // fixed effects priors
    fixed_effects ~ normal(0.0, 1.0);

    // variable effect hyperpriors
    shortstop_effect_mean ~ normal(0.0, 1.0);
    shortstop_effect_var ~ exponential(1.0);
    // shortstop effect prior
    shortstop_effect ~ normal(shortstop_effect_mean, shortstop_effect_var);
    
    // output 
    labels ~ bernoulli_logit(output);
}