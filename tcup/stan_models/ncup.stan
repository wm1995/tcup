// This is a prototype of a parallel model to t-cup using normal distributions

data {
    int<lower=0> N;                 // Number of data points
    int<lower=1> K;                 // Number of independent vars
    array[N] real y;                // Dependent variable
    array[N] real<lower=0> dy;      // Err. in dependent variable
    array[N] vector[K] x;           // Independent variables
    array[N] vector<lower=0>[K] dx; // Err. in independent vars
    array[N] corr_matrix[K] rho;    // Correlation matrix btwn indep. vars
}

parameters {
    // True values:
    array[N] real true_y;  // Transformed y values
    array[N] vector[K] true_x;  // Transformed x values

    // Regression coefficients:
    real alpha;                  // Intercept
    vector[K] beta;              // x coefficients
    real<lower=0> sigma;
}

model {
    // Model
    for(n in 1:N){
        true_y[n] ~ normal(alpha + beta .* true_x[n], sigma);
        x[n] ~ normal(true_x[n], dx[n]);
        y[n] ~ normal(true_y[n], dy[n]);
    }

    alpha ~ normal(0, 3);
    for(k in 1:K)
        beta[k] ~ normal(0, 3);
    sigma ~ cauchy(0, 1);
}
