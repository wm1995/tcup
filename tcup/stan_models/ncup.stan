// This is a prototype of a parallel model to t-cup using normal distributions

data {
    int<lower=0> N;                 // Number of data points
    int<lower=1> D;                 // Number of independent vars
    array[N] real y;                // Dependent variable
    array[N] real<lower=0> dy;      // Err. in dependent variable
    array[N] vector[D] x;           // Independent variables
    array[N] cov_matrix[D] dx; // Err. in independent vars

    real shape_param; // Defined for pipeline compatibility but does nothing
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
        x[n] ~ multi_normal(true_x[n], dx[n]);
        y[n] ~ normal(true_y[n], dy[n]);
    }

    alpha ~ normal(0, 3);
    for(d in 1:D)
        beta[d] ~ normal(0, 3);
    sigma ~ cauchy(0, 1);
}
