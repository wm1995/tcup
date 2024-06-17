import pytest

import arviz as az
import jax
import jax.numpy as jnp
from numpyro.distributions import Normal
from numpyro.infer import Predictive
import scipy.stats as sps
from tcup.numpyro import model_builder, tcup


THRESHOLD = 1e-3  # Set the p-value threshold


@pytest.fixture
def tcup_samples():
    rng_key = jax.random.PRNGKey(0)
    tcup_model = model_builder(Normal())
    tcup_sampler = Predictive(tcup_model, num_samples=100000)
    return tcup_sampler(
        rng_key,
        x_scaled=jnp.array([[0]]),
        y_scaled=jnp.array([[0]]),
        cov_x_scaled=jnp.array([[[1]]]),
        dy_scaled=jnp.array([[1]]),
    )


@pytest.fixture
def ncup_samples():
    rng_key = jax.random.PRNGKey(0)
    ncup_model = model_builder(Normal(), ncup=True)
    ncup_sampler = Predictive(ncup_model, num_samples=100000)
    return ncup_sampler(
        rng_key,
        x_scaled=jnp.array([[0]]),
        y_scaled=jnp.array([[0]]),
        cov_x_scaled=jnp.array([[[1]]]),
        dy_scaled=jnp.array([[1]]),
    )


@pytest.fixture
def fixed3_samples():
    rng_key = jax.random.PRNGKey(0)
    fixed3_model = model_builder(Normal(), fixed_nu=3)
    fixed3_sampler = Predictive(fixed3_model, num_samples=100000)
    return fixed3_sampler(
        rng_key,
        x_scaled=jnp.array([[0]]),
        y_scaled=jnp.array([[0]]),
        cov_x_scaled=jnp.array([[[1]]]),
        dy_scaled=jnp.array([[1]]),
    )


def test_tcup_intrinsic_dist(tcup_samples):
    x = tcup_samples["x_true"].flatten()
    y = tcup_samples["y_true"].flatten()
    alpha = tcup_samples["alpha_scaled"].flatten()
    beta = tcup_samples["beta_scaled"].flatten()
    sigma = tcup_samples["sigma_scaled"].flatten()
    nu = tcup_samples["nu"].flatten()

    mu = alpha + jnp.multiply(x, beta)
    t = (y - mu) / sigma

    assert sps.kstest(sps.t.cdf(t, df=nu), sps.uniform.cdf).pvalue > THRESHOLD
    assert sps.kstest(t, sps.norm.cdf).pvalue < THRESHOLD


def test_ncup_intrinsic_dist(ncup_samples):
    x = ncup_samples["x_true"].flatten()
    y = ncup_samples["y_true"].flatten()
    alpha = ncup_samples["alpha_scaled"].flatten()
    beta = ncup_samples["beta_scaled"].flatten()
    sigma = ncup_samples["sigma_scaled"].flatten()

    mu = alpha + jnp.multiply(x, beta)
    z = (y - mu) / sigma

    assert sps.kstest(z, sps.norm.cdf).pvalue > THRESHOLD


def test_fixed3_intrinsic_dist(fixed3_samples):
    x = fixed3_samples["x_true"].flatten()
    y = fixed3_samples["y_true"].flatten()
    alpha = fixed3_samples["alpha_scaled"].flatten()
    beta = fixed3_samples["beta_scaled"].flatten()
    sigma = fixed3_samples["sigma_scaled"].flatten()

    mu = alpha + jnp.multiply(x, beta)
    t = (y - mu) / sigma

    assert sps.kstest(t, sps.t(df=3).cdf).pvalue > THRESHOLD
    assert sps.kstest(t, sps.norm.cdf).pvalue < THRESHOLD


def test_tcup(data):
    mcmc = tcup(**data)
    assert isinstance(mcmc, az.InferenceData)
