import arviz as az
from tcup.numpyro import tcup


def test_tcup(outlier_data):
    mcmc = tcup(outlier_data)
    assert isinstance(mcmc, az.InferenceData)
