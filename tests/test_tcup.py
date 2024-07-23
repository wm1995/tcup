import arviz as az
from jax.config import config

import tcup

config.update("jax_enable_x64", True)


def test_tcup(data):
    mcmc = tcup.tcup(**data)
    assert isinstance(mcmc, az.InferenceData)


def test_ncup(data):
    mcmc = tcup.tcup(**data, model="ncup")
    assert isinstance(mcmc, az.InferenceData)


def test_fixed(data):
    mcmc = tcup.tcup(**data, model="fixed", shape_param=3)
    assert isinstance(mcmc, az.InferenceData)
