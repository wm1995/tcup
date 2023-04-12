import arviz as az
from jax.config import config
import tcup

config.update("jax_enable_x64", True)


def test_tcup(data):
    mcmc = tcup.tcup(data)
    assert isinstance(mcmc, az.InferenceData)
