import arviz as az
import pytest
from tcup.stan import tcup


@pytest.mark.parametrize("model", ["tcup", "ncup"])
def test_tcup(outlier_data, model):
    mcmc = tcup(outlier_data, model=model)
    assert isinstance(mcmc, az.InferenceData)


@pytest.mark.parametrize(
    "prior",
    [
        "invgamma",
        "invgamma2",
        "cauchy",
        "cauchy_scaled",
        "cauchy_truncated",
        "F18",
        "F18reparam",
        "nu2",
        "nu2_principled",
        "nu2_heuristic",
        "nu2_scaled",
        "invnu",
    ],
)
def test_priors(outlier_data, prior):
    mcmc = tcup(outlier_data, model="tcup", prior=prior)
    assert isinstance(mcmc, az.InferenceData)
