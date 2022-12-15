// This is a prototype of the t-cup code, with all scaling removed

functions {
    // Estimate shape parameter that gives peak height t
    // This is an approximation to the true functional form
    real shape_func(real t){
        real nu;
        nu = 1 / cos(pi() / 2 * sqrt(t));
        return nu;
    }
}

data {
    int<lower=0> N;                 // Number of data points
    int<lower=1> K;                 // Number of independent vars
    array[N] real y;                // Dependent variable
    array[N] real<lower=0> dy;      // Err. in dependent variable
    array[N] vector[K] x;           // Independent variables
    array[N] vector<lower=0>[K] dx; // Err. in independent vars
    array[N] corr_matrix[K] rho;    // Correlation matrix btwn indep. vars

    // Shape parameter for Student-t distribution
    // Should be > 0 (or use -1 to add this as a parameter to be learned)
    real shape_param;
}

transformed data {
    // If True, then learn Student-t shape parameter
    int shape_param_flag = (shape_param == -1);
    array[N] cov_matrix[K] cov_x;

    for (n in 1:N){
        cov_x[n] = diag_post_multiply(diag_pre_multiply(dx[n], rho[n]), dx[n]);
    }
}

parameters {
    // True values:
    array[N] real true_y;  // Transformed y values
    array[N] vector[K] true_x;  // Transformed x values

    // Regression coefficients:
    real alpha;                  // Intercept
    vector[K] beta;              // x coefficients
    real<lower=0> sigma;

    // Distribution parameters:
    real<lower=0, upper=1> peak_height;
}

model {
    // Student-t shape parameter:
    real nu;
    if(!shape_param_flag)
        nu = shape_param;
    else
        nu = shape_func(peak_height);

    // Model
    for(n in 1:N){
        true_y[n] ~ student_t(nu, alpha + beta .* true_x[n], sigma);
        x[n] ~ multi_student_t(nu, true_x[n], cov_x[n]);
        y[n] ~ student_t(nu, true_y[n], dy[n]);
    }

    alpha ~ normal(0, 3);
    for(k in 1:K)
        beta[k] ~ normal(0, 3);
    sigma ~ cauchy(0, 1);
    peak_height ~ uniform(0, 1);
}
