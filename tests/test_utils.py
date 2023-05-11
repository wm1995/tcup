from jax.config import config
import numpy as np
import pytest
import scipy.stats as sps
from tcup.utils import outlier_frac, sigma_68

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("nu", [0.1, 0.3, 1, 2, 5, 1000])
def test_outlier_frac(nu):
    frac = 2 * sps.t.cdf(-3, nu)
    assert np.isclose(outlier_frac(nu), frac)


@pytest.mark.parametrize("nu", [0.1, 0.3, 0.5, 1, 2, 4, 10, 100, 1000])
def test_sigma_68(nu):
    sigma = sigma_68(nu)
    assert np.isclose(sps.t.cdf(sigma, nu), sps.norm.cdf(1))
