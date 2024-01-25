import arviz as az
from tcup.numpyro import tcup


def test_tcup(data):
    mcmc = tcup(**data)
    assert isinstance(mcmc, az.InferenceData)
