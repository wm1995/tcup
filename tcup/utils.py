import jax
import jax.numpy as jnp
import jax.scipy.special as jspec
import tensorflow_probability.substrates.jax.math as tfp_math


@jax.jit
def peak_height(nu):
    log_t = 0.5 * jnp.log(2)
    log_t -= 0.5 * jnp.log(nu)
    log_t += jspec.gammaln((nu + 1) / 2)
    log_t -= jspec.gammaln(nu / 2)
    t = jnp.where(nu == 0.0, 0.0, jnp.exp(log_t))
    t = jnp.where(jnp.isinf(nu), 1.0, t)
    return t


@jax.jit
def normality(nu):
    t = peak_height(nu)
    c = peak_height(1)  # Cauchy peak height
    return (t - c) / (1 - c)


@jax.jit
def t_cdf(nu, x):
    # Using https://encyclopediaofmath.org/wiki/Student_distribution
    # The above has a source for this in terms of the incomplete Beta function
    # It's only for x > 0 so I modified it to work for all x
    # If x < 0, return integral
    # If x == 0, return 1/2
    # If x > 0, return 1 - integral
    integral = 0.5 * tfp_math.betainc(0.5 * nu, 0.5, nu / (nu + x**2))
    return (jnp.sign(x) + 1) / 2 - jnp.sign(x) * integral


@jax.jit
def outlier_frac(nu, outlier_sigma=3):
    normal_outlier_frac = 1 - jspec.erf(outlier_sigma / jnp.sqrt(2))
    omega = tfp_math.betainc(0.5 * nu, 0.5, nu / (nu + outlier_sigma**2))
    omega = jnp.where(nu == 0, 0.0, omega)
    omega = jnp.where(jnp.isinf(nu), normal_outlier_frac, omega)
    return omega


@jax.jit
def sigma_68(nu):
    normal_outlier_frac = 1 - jspec.erf(jnp.sqrt(0.5))
    frac = tfp_math.betaincinv(0.5 * nu, 0.5, normal_outlier_frac)
    sigma_68 = jnp.sqrt(nu / frac - nu)
    return sigma_68
