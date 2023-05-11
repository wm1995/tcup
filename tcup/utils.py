import contextlib
import sys

import jax
import jax.numpy as jnp
import jax.scipy.special as jspec
import tensorflow_probability.substrates.jax.math as tfp_math


# The following class and context manager are for suppressing stdout
# This code is taken from https://stackoverflow.com/a/2829036
class DummyFile(object):
    def write(self, x):
        pass


@contextlib.contextmanager
def suppress_output():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


@jax.jit
def sigma_68(nu):
    normal_outlier_frac = 1 - jspec.erf(jnp.sqrt(0.5))
    frac = tfp_math.betaincinv(0.5 * nu, 0.5, normal_outlier_frac)
    sigma_68 = jnp.sqrt(nu / frac - nu)
    return sigma_68


@jax.jit
def outlier_frac(nu, outlier_sigma=3):
    normal_outlier_frac = 1 - jspec.erf(outlier_sigma / jnp.sqrt(2))
    omega = tfp_math.betainc(0.5 * nu, 0.5, nu / (nu + outlier_sigma**2))
    omega = jnp.where(nu == 0, 0.0, omega)
    omega = jnp.where(jnp.isinf(nu), normal_outlier_frac, omega)
    return omega
