import arviz as az
from jax.config import config
import pytest
from tcup.stan import tcup

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["tcup", "ncup"])
def test_tcup(outlier_data, model):
    mcmc = tcup(outlier_data, model=model)
    assert isinstance(mcmc, az.InferenceData)
