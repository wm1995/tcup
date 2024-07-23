// t-cup with Gaussian mixture prior
// Jinja2 template

{% if reparam %}
functions {
{{ reparam.functions }}
}
{% endif %}

data {
    int<lower=0> N;            // Number of data points
    int<lower=1> D;            // Number of independent vars
    array[N] real y_scaled;           // Dependent variable
    array[N] real<lower=0> dy_scaled; // Err. in dependent variable
    array[N] vector[D] x_scaled;      // Independent variables
    array[N] cov_matrix[D] cov_x_scaled; // Err. in independent vars

{% if model == "fixed" %}
    // Shape parameter for Student's t-distribution
    real<lower=0> nu;
{% endif %}

    // Gaussian mixture prior
    int<lower=1> K;                    // Components
    simplex[K] theta_mix;              // Mixing proportions
    array[K] vector[D] mu_mix;         // Locations of mixture components
    array[K] cov_matrix[D] sigma_mix;  // Scales of mixture components
}

parameters {
    // True values
    array[N] vector[D] true_x;   // Latent x values

    // Regression coefficients
    real alpha_scaled;                  // Intercept
    vector[D] beta_scaled_tsfrm;              // x coefficients
    real<lower=0> sigma_scaled;

    // true_y
{% if model == "ncup" %}
    array[N] real epsilon; // Scatter for each datapoint
{% else %}
    // Mixture model reparameterisation of t distribution
    array[N] real epsilon_tsfrm;
    array[N] real<lower=0> tau_epsilon;
{% endif %}

{% if model == "tcup" %}
    // Distribution parameters
    {% if reparam %}
    {{ reparam.params }}
    {% else %}
    real<lower=0> nu;
    {% endif %}
{% endif %}
}

transformed parameters {
    // Transformed parameters
    array[N] real true_y;

    vector[D] beta_scaled = tan(beta_scaled_tsfrm);

    {% if reparam %}
    {{ reparam.transformed_params }}
    {% endif %}

{% if model == "ncup" %}
    for(n in 1:N){
        true_y[n] = alpha_scaled + dot_product(beta_scaled, true_x[n]) + epsilon[n];
    }
{% else %}
    // Mixture model reparameterisation of t distribution
    array[N] real epsilon;
    for(n in 1:N){
        epsilon[n] = sigma_scaled * epsilon_tsfrm[n] / sqrt(tau_epsilon[n]);
        true_y[n] = alpha_scaled + dot_product(beta_scaled, true_x[n]) + epsilon[n];
    }
{% endif %}
}

model {
{% if model == "ncup" %}
    // Model
    for(n in 1:N){
        epsilon[n] ~ std_normal();
    }
{% else %}
    // t-distribution shape parameter
    {% if reparam %}
    {{ reparam.half_nu }}
    {% else %}
    real half_nu = nu / 2;
    {% endif %}

    // Model
    for(n in 1:N){
        // Equivalent to true_y ~ student_t(nu, alpha_scaled + beta_scaled . true_x[n], sigma_scaled);
        epsilon_tsfrm[n] ~ std_normal();
        tau_epsilon[n] ~ gamma(half_nu, half_nu);
    }
{% endif %}
    // Model
    for(n in 1:N){
        x_scaled[n] ~ multi_normal(true_x[n], cov_x_scaled[n]);
        y_scaled[n] ~ normal(true_y[n], dy_scaled[n]);
    }

    // Prior
    alpha_scaled ~ normal(0, 3);
    for(d in 1:D)
        beta_scaled_tsfrm[d] ~ uniform(-pi() / 2, pi() / 2);
    sigma_scaled ~ {{ sigma_prior | default("gamma(2, 2)") }};
{% if model == "tcup" %}
    {% if reparam %}
    {{ reparam.prior }}
    {% else %}
    nu ~ {{ nu_prior | default("inv_gamma(3, 10)") }};
    {% endif %}
{% endif %}

    // Gaussian mixture prior
    vector[K] log_theta = log(theta_mix);  // cache log calculation
    for (n in 1:N) {
        vector[K] lps = log_theta;
        for (k in 1:K) {
            lps[k] += multi_normal_lpdf(true_x[n] | mu_mix[k], sigma_mix[k]);
        }
        target += log_sum_exp(lps);
    }
}
