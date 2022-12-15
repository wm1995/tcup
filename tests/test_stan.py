import arviz as az
import pytest
from tcup.stan import tcup


@pytest.mark.parametrize("model", ["tcup", "ncup"])
def test_tcup(outlier_data, model):
    mcmc = tcup(outlier_data, model=model)
    assert isinstance(mcmc, az.InferenceData)
