// n-cup with Gaussian mixture prior

data {
    int<lower=0> N;                 // Number of data points
    int<lower=1> D;                 // Number of independent vars
    array[N] real y;                // Dependent variable
    array[N] real<lower=0> dy;      // Err. in dependent variable
    array[N] vector[D] x;           // Independent variables
    array[N] cov_matrix[D] cov_x; // Err. in independent vars

    real shape_param; // Defined for pipeline compatibility but does nothing

    // Gaussian mixture prior
    int<lower=1> K;                    // Components
    simplex[K] theta_mix;              // Mixing proportions
    array[K] vector[D] mu_mix;         // Locations of mixture components
    array[K] cov_matrix[D] sigma_mix;  // Scales of mixture components
}

parameters {
    // True values:
    array[N] real true_y;  // Transformed y values
    array[N] vector[D] true_x;  // Transformed x values

    // Regression coefficients:
    real alpha;                  // Intercept
    vector[D] beta;              // x coefficients
    real<lower=0> sigma;
}

model {
    // Model
    for(n in 1:N){
        true_y[n] ~ normal(alpha + beta .* true_x[n], sigma);
        x[n] ~ multi_normal(true_x[n], cov_x[n]);
        y[n] ~ normal(true_y[n], dy[n]);
    }

    alpha ~ normal(0, 3);
    for(d in 1:D)
        beta[d] ~ normal(0, 3);
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
