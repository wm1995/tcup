from jax.config import config
import numpy as np
import pytest
import scipy.stats as sps
from tcup.utils import sigma_68

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("nu", [0.1, 0.3, 0.5, 1, 2, 4, 10, 100, 1000])
def test_sigma_68(nu):
    sigma = sigma_68(nu)
    assert np.isclose(sps.t.cdf(sigma, nu), sps.norm.cdf(1))
