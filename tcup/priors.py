from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy.special as jspec
import tensorflow_probability.substrates.jax.distributions as tfp_stats

from .utils import peak_height


@jax.jit
def pdf_invgamma(nu):
    ALPHA = 3
    BETA = 10
    return tfp_stats.InverseGamma(ALPHA, BETA).prob(nu)


@jax.jit
def pdf_invgamma2(nu):
    ALPHA = 2
    BETA = 6
    return tfp_stats.InverseGamma(ALPHA, BETA).prob(nu)


@jax.jit
def pdf_F18(nu):
    a = 1.2
    nu_0 = 0.55
    P_nu = ((nu / nu_0) ** (1.0 / 2.0 / a) + (nu / nu_0) ** (2.0 / a)) ** -a
    log_norm = (
        jspec.gammaln(a / 3) + jspec.gammaln(2 * a / 3) - jspec.gammaln(a)
    )
    norm = 2 / 3 * a * nu_0 * jnp.exp(log_norm)
    return P_nu / norm


@partial(jnp.vectorize, excluded={1})
def pdf_F18reparam(nu, coord):
    # P(x) = P(t) dt / dx
    # t ~ U(0, 1)
    @jax.jit
    def nu_approx(t):
        a = 4.747
        b = 1.443
        alpha = (0.125 * t / (1 - t)) * jnp.exp(-a * (1 - t) ** b)
        alpha += t**2 / jnp.pi
        return 2 * alpha

    @jax.jit
    def t_approx(x):
        t_interp = jnp.linspace(0, 1, 100000)
        x_interp = coord(nu_approx(t_interp))
        ind = jnp.argsort(x_interp)
        return jnp.interp(x, x_interp[ind], t_interp[ind], right=1)

    dt_approx = jax.grad(t_approx)
    x = coord(nu)
    return jnp.abs(dt_approx(x))


@jnp.vectorize
def pdf_peak_height(nu, nu_min=0.0, nu_max=jnp.inf):
    # P(nu) = P(t) dt / d_nu
    # t ~ U(t(nu_min), t(nu_max))
    norm = 1 / (peak_height(nu_max) - peak_height(nu_min))
    dt = jax.grad(peak_height)
    P_nu = jnp.where(
        jnp.logical_and(nu > nu_min, nu < nu_max),
        norm * dt(nu),
        0.0,
    )
    return P_nu


@jax.jit
def pdf_inv_nu(nu):
    # P(nu) = P(theta) |dtheta / d_nu|
    # 1/nu = theta ~ U(0, 1)
    # I've already taken the absolute value below
    dtheta = 1 / nu**2
    return jnp.where(nu >= 1, dtheta, 0.0)


@jnp.vectorize
def pdf_cauchy(nu):
    # P(nu) = P(t) dt / d_nu
    # t ~ U(t(nu = 1), 1)
    @jax.jit
    def nu_approx(t):
        return 1 / jnp.cos(jnp.pi / 2 * jnp.sqrt(t))

    t = peak_height(nu)
    d_nu = jax.grad(nu_approx)
    P_nu = jnp.where(nu >= 1, 1 / d_nu(t), 0.0)
    return P_nu


@jnp.vectorize
def pdf_nu2(nu):
    # P(nu) = P(t) dt / d_nu
    # t ~ U(t(nu = 2), 1)
    @jax.jit
    def nu_approx(t):
        return 2 / jnp.cos(jnp.pi / 2 * jnp.sqrt(t))

    t = peak_height(nu)
    d_nu = jax.grad(nu_approx)
    P_nu = jnp.where(nu >= 2, 1 / d_nu(t), 0.0)
    return P_nu
