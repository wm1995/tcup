// t-cup
// Prior: Feeney et al. 2018

functions {
    // Prior from Feeney et al. 2018
    // Approximates uniform peak height from uniform to normal
    real log_p_nu(real nu) {
        real nu_0 = 0.55;
        real a = 1.2;
        real log_nu = log(nu) - log(nu_0);
        return -a * log_sum_exp(log_nu / (2 * a), log_nu * (2 / a));
    }
}

data {
    int<lower=0> N;            // Number of data points
    int<lower=1> K;            // Number of independent vars
    array[N] real y;           // Dependent variable
    array[N] real<lower=0> dy; // Err. in dependent variable
    array[N] vector[K] x;      // Independent variables
    array[N] cov_matrix[K] dx; // Err. in independent vars

    // Shape parameter for Student's t-distribution
    // Should be > 0 (or use -1 to add this as a parameter to be learned)
    real shape_param;
}

transformed data {
    // If True, then learn t-distribution shape parameter
    int shape_param_flag = (shape_param == -1);
}

parameters {
    // True values
    array[N] vector[K] true_x;   // Latent x values

    // Regression coefficients
    real alpha;                  // Intercept
    vector[K] beta;              // x coefficients

    // Transformed parameters
    // sigma
    real<lower=0, upper=pi()/2> sigma_tsfrm;
    // true_y
    array[N] real epsilon_tsfrm; // Scatter for each datapoint
    array[N] real<lower=0> tau_epsilon; // Scatter for each datapoint

    // Distribution parameters
    real<lower=0> nu;
}

transformed parameters {
    // Transformed parameters
    real sigma = tan(sigma_tsfrm);
    array[N] real true_y;

    array[N] real epsilon;
    for(n in 1:N){
        epsilon[n] = sigma * epsilon_tsfrm[n];
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

        x[n] ~ multi_student_t(nu, true_x[n], dx[n]);
        y[n] ~ student_t(nu, true_y[n], dy[n]);
    }

    // Prior
    alpha ~ normal(0, 3);
    for(k in 1:K)
        beta[k] ~ normal(0, 3);
    target += log_p_nu(nu);
    // Equivalent to sigma ~ half_cauchy(0, 1);
    sigma_tsfrm ~ uniform(0, pi() / 2);
}
