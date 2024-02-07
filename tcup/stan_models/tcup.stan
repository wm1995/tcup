// t-cup with Gaussian mixture prior

data {
    int<lower=0> N;            // Number of data points
    int<lower=1> D;            // Number of independent vars
    array[N] real y;           // Dependent variable
    array[N] real<lower=0> dy; // Err. in dependent variable
    array[N] vector[D] x;      // Independent variables
    array[N] cov_matrix[D] cov_x; // Err. in independent vars

    // Shape parameter for Student's t-distribution
    // Should be > 0 (or use -1 to add this as a parameter to be learned)
    real shape_param;

    // Gaussian mixture prior
    int<lower=1> K;                    // Components
    simplex[K] theta_mix;              // Mixing proportions
    array[K] vector[D] mu_mix;         // Locations of mixture components
    array[K] cov_matrix[D] sigma_mix;  // Scales of mixture components
}

transformed data {
    // If True, then learn t-distribution shape parameter
    int shape_param_flag = (shape_param == -1);
}

parameters {
    // True values
    array[N] vector[D] true_x;   // Latent x values

    // Regression coefficients
    real alpha;                  // Intercept
    vector[D] beta;              // x coefficients
    real<lower=0> sigma;

    // true_y
    array[N] real epsilon_tsfrm; // Scatter for each datapoint
    array[N] real<lower=0> tau_epsilon; // Scatter for each datapoint

    // Distribution parameters
    real<lower=0> nu;
}

transformed parameters {
    // Transformed parameters
    array[N] real true_y;

    array[N] real epsilon;
    for(n in 1:N){
        epsilon[n] = sigma * epsilon_tsfrm[n] / sqrt(tau_epsilon[n]);
        true_y[n] = alpha + dot_product(beta, true_x[n]) + epsilon[n];
    }
}

model {
    // t-distribution shape parameter
    real nu_;
    if(!shape_param_flag)
        nu_ = shape_param;
    else
        nu_ = nu;
    real half_nu = nu_ / 2;

    // Model
    for(n in 1:N){
        // Equivalent to true_y ~ student_t(nu, alpha + beta . true_x[n], sigma);
        epsilon_tsfrm[n] ~ std_normal();
        tau_epsilon[n] ~ gamma(half_nu, half_nu);

        x[n] ~ multi_normal(true_x[n], cov_x[n]);
        y[n] ~ normal(true_y[n], dy[n]);
    }

    // Prior
    alpha ~ normal(0, 3);
    for(d in 1:D)
        beta[d] ~ normal(0, 3);
    nu ~ inv_gamma(3, 10);
    sigma ~ gamma(2, 2);

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
