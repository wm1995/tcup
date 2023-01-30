import jax
import jax.numpy as jnp
import jax.scipy.special as jspec


@jax.jit
def peak_height(nu):
    log_t = 0.5 * jnp.log(2)
    log_t -= 0.5 * jnp.log(nu)
    log_t += jspec.gammaln((nu + 1) / 2)
    log_t -= jspec.gammaln(nu / 2)
    return jnp.exp(log_t)


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
    cdf = jspec.betainc(0.5 * nu, 0.5, nu / (nu + x**2)) - 1
    cdf *= jnp.sign(x)
    return (1 - cdf) / 2


@jax.jit
def outlier_frac(nu):
    OUTLIER_SIGMA = 3
    return jspec.betainc(0.5 * nu, 0.5, nu / (nu + OUTLIER_SIGMA**2))
