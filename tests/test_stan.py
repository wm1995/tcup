import arviz as az
from tcup.stan import ncup, tcup


def test_ncup(outlier_data):
    mcmc = ncup(outlier_data)
    assert isinstance(mcmc, az.InferenceData)


def test_tcup(outlier_data):
    mcmc = tcup(outlier_data)
    assert isinstance(mcmc, az.InferenceData)
