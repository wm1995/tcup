from jax.config import config
import numpy as np
import pytest
import scipy.stats as sps
from tcup.utils import peak_height, normality, outlier_frac, t_cdf, sigma_68

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("nu", [0.1, 0.3, 1, 2, 5, 1000])
def test_outlier_frac(nu):
    frac = 2 * sps.t.cdf(-3, nu)
    assert np.isclose(outlier_frac(nu), frac)


@pytest.mark.parametrize("nu", [0.1, 0.3, 0.5, 1, 2, 4, 10, 100, 1000])
@pytest.mark.parametrize("x", [-3, -1, 0, 1, 2, 3, 10])
def test_t_cdf(nu, x):
    assert np.isclose(t_cdf(nu, x), sps.t.cdf(x, nu)).all()


@pytest.mark.parametrize("nu", [0.1, 0.3, 1, 2, 5, 1000])
def test_peak_height(nu):
    t = sps.t.pdf(0, nu) / sps.norm.pdf(0)
    assert np.isclose(peak_height(nu), t)


@pytest.mark.parametrize("nu", [1, 1.5, 2, 3, 10, 1000])
def test_normality(nu):
    t = sps.t.pdf(0, nu) - sps.cauchy.pdf(0)
    t /= sps.norm.pdf(0) - sps.cauchy.pdf(0)
    assert np.isclose(normality(nu), t)


@pytest.mark.parametrize("nu", [0.1, 0.3, 0.5, 1, 2, 4, 10, 100, 1000])
def test_sigma_68(nu):
    sigma = sigma_68(nu)
    assert np.isclose(sps.t.cdf(sigma, nu), sps.norm.cdf(1))
