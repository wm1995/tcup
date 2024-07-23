import arviz as az
import pytest
from jax import config

from tcup.stan import tcup

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["tcup", "ncup"])
def test_tcup(data, model):
    mcmc = tcup(**data, model=model)
    assert isinstance(mcmc, az.InferenceData)
