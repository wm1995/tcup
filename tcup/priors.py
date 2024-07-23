from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jspec
import tensorflow_probability.substrates.jax.distributions as tfp_stats

from .utils import peak_height


def pdf_peak_height(nu, coord, nu_min=0.0, nu_max=jnp.inf):
    @partial(jnp.vectorize, excluded={1, 2, 3})
    def pdf(nu, coord, nu_min, nu_max):
        # P(nu) = P(t) dt / d_nu
        # t ~ U(t(nu_min), t(nu_max))
        norm = 1 / (peak_height(nu_max) - peak_height(nu_min))
        dt = jax.grad(peak_height)
        dx = jax.grad(coord)
        P_nu = jnp.where(
            jnp.logical_and(nu > nu_min, nu < nu_max),
            norm * jnp.abs(dt(nu) / dx(nu)),
            0.0,
        )
        return P_nu

    return pdf(nu, coord, nu_min, nu_max)


def pdf_inv_nu(nu, coord):
    grad_x = jnp.vectorize(jax.grad(coord))
    # P(nu) = P(theta) |dtheta / d_nu|
    # 1/nu = theta ~ U(0, 1)
    # I've already taken the absolute value below
    dtheta = 1 / nu**2
    if coord == peak_height:
        # The following is a weird hack for peak height only
        # In the limit where nu is large, we can end up with numerical errors
        # This leads to the true probability being underestimated
        # Therefore, let's clip at the limiting value
        P_nu = jnp.where(
            nu >= 1,
            jnp.clip(dtheta / jnp.abs(grad_x(nu)), a_min=4.0),
            0.0,
        )
    else:
        P_nu = jnp.where(nu >= 1, dtheta, 0.0) / jnp.abs(grad_x(nu))

    return P_nu


def pdf_invgamma(nu, coord):
    ALPHA = 3
    BETA = 10
    grad_x = jnp.vectorize(jax.grad(coord))
    return tfp_stats.InverseGamma(ALPHA, BETA).prob(nu) / jnp.abs(grad_x(nu))


def pdf_invgamma2(nu, coord):
    ALPHA = 2
    BETA = 6
    grad_x = jnp.vectorize(jax.grad(coord))
    return tfp_stats.InverseGamma(ALPHA, BETA).prob(nu) / jnp.abs(grad_x(nu))


def pdf_F18(nu, coord):
    a = 1.2
    nu_0 = 0.55
    grad_x = jnp.vectorize(jax.grad(coord))

    P_nu = ((nu / nu_0) ** (1.0 / 2.0 / a) + (nu / nu_0) ** (2.0 / a)) ** -a
    log_norm = (
        jspec.gammaln(a / 3) + jspec.gammaln(2 * a / 3) - jspec.gammaln(a)
    )
    norm = 2 / 3 * a * nu_0 * jnp.exp(log_norm)
    return P_nu / norm / jnp.abs(grad_x(nu))


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


@partial(jnp.vectorize, excluded={1, 2})
def _pdf_cos_approx(nu, coord, nu_min):
    # P(x) = P(t) dt / d_nu d_nu / dx
    # t ~ U(t(nu = nu_min), 1)
    @jax.jit
    def t_approx(nu):
        scaled_t_approx = (2 / jnp.pi * jnp.arccos(nu_min / nu)) ** 2
        return (scaled_t_approx - 1) * (1 - peak_height(nu_min)) + 1

    norm = 1 / (t_approx(jnp.inf) - t_approx(nu_min))

    grad_t_approx = jax.grad(t_approx)
    grad_x = jax.grad(coord)
    prob = norm * jnp.abs(grad_t_approx(nu) / grad_x(nu))
    prob = jnp.where(nu >= nu_min, prob, 0.0)

    # The below is another numerical hack for peak height
    if coord == peak_height:
        # This is a really quick and dirty hack
        # 16 / pi is the limit as calculated by Mathematica
        # The numerical issues kick in for nu > ~1e5
        # Can't clip as before because it increases then decreases again
        prob = jnp.where(nu < 1e5, prob, nu_min * 16 / jnp.pi)
    return prob


def pdf_cauchy(nu, coord):
    # For this prior, nu_min = 1
    return _pdf_cos_approx(nu, coord, 1)


def pdf_nu2(nu, coord):
    # For this prior, nu_min = 2
    return _pdf_cos_approx(nu, coord, 2)
