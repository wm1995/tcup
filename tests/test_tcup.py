import arviz as az
from jax.config import config
import tcup

config.update("jax_enable_x64", True)


def test_tcup(outlier_data):
    mcmc = tcup.tcup(outlier_data)
    assert isinstance(mcmc, az.InferenceData)
