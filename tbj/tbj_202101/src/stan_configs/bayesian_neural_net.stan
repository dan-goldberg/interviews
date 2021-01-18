
data {
    // neural network input
    int<lower=1> num_samples; // number of pitches
    int<lower=1> num_features; // number of pitch features
    matrix[num_samples, num_features] feature_vectors; // data matrix
    
    // neural network config
    int<lower=0> num_hidden1_nodes; // number of nodes in hidden layer 1
    
    // cluster groups for variable effects
    // int<lower=0> num_pitchers // number of pitchers
    // int<lower=0> num_catchers // number of catchers
    // int<lower=0> num_umpires // number of umpires
    // int<lower=0> num_batters // number of batters
    
    // output
    int<lower=0,upper=1> labels[num_samples]; // label 1 if strike, 0 if ball
}
parameters {
    // neural network params
    matrix[num_features, num_hidden1_nodes] layer01_weights; // input-to-layer-1 weights
    vector[num_hidden1_nodes] layer01_bias; // input-to-layer-1 bias
    
    vector[num_hidden1_nodes] output_weights; // layer-1-to-output weights
    real output_bias; // layer-1-to-output bias
    
    // variable effects params
    // vector[num_pitchers] pitcher_effect_mean;
    // vector[num_pitchers]<lower=0> pitcher_effect_var;
    
    // vector[num_batters] batter_effect_mean;
    // vector[num_batters]<lower=0> batter_effect_var;
    
    // vector[num_catchers] catcher_effect_mean;
    // vector[num_catchers]<lower=0> catcher_effect_var;
    
    // vector[num_umpires] umpire_effect_mean;
    // vector[num_umpires]<lower=0> umpire_effect_var;
}
transformed parameters {
    matrix[num_samples, num_hidden1_nodes] hidden1_nodes; // activation values of layer1 for each datapoint
    vector[num_samples] output_raw; // output before variable effects and link
    // vector[num_samples] output; // output after variable effects before link 

    hidden1_nodes = tanh(feature_vectors * layer01_weights + rep_matrix(layer01_bias', num_samples));
    output_raw = hidden1_nodes * output_weights + output_bias;
    // output = //
}
model {
    // priors
    for (i in 1:num_features) {
        layer01_weights[i, ] ~ normal(0.0, 1.0);
    }
    layer01_bias ~ normal(0.0, 0.5);
    for (i in 1:num_hidden1_nodes) {
        output_weights[i] ~ normal(0.0, 1.0);
    }
    output_bias ~ normal(0.0, 0.5);
    
    // output 
    labels ~ bernoulli_logit(output_raw);
}