import importlib.resources as pkg_resources
import arviz as az
import numpy as np
import stan

from . import stan_models


def _get_model_src(model, prior):
    if prior is None:
        if model == "ncup":
            return pkg_resources.read_text(stan_models, f"{model}.stan")
        else:
            prior = "cauchy"

    if model == "ncup":
        raise ValueError("No choice of prior with ncup model")

    return pkg_resources.read_text(stan_models, f"{model}_{prior}.stan")


def _prep_data(data):
    # If nu is not provided, set to -1 to infer as part of model
    shape_param = data.get("nu", -1)

    # Extract data shape
    match data["x"].shape:
        case (N, K):
            stan_data = {
                "N": N,
                "K": K,
                "x": data["x"].tolist(),
                "dx": data["dx"].tolist(),
                "y": data["y"].tolist(),
                "dy": data["dy"].tolist(),
                "shape_param": shape_param,
            }
        case (N,):
            stan_data = {
                "N": N,
                "K": 1,
                "x": data["x"][:, np.newaxis].tolist(),
                "dx": data["dx"][:, np.newaxis, np.newaxis].tolist(),
                "y": data["y"].tolist(),
                "dy": data["dy"].tolist(),
                "shape_param": shape_param,
            }

    return stan_data


def tcup(data, seed=None, model="tcup", prior=None, **sampler_kwargs):
    stan_data = _prep_data(data)

    model_src = _get_model_src(model, prior)

    sampler = stan.build(model_src, stan_data, random_seed=seed)

    sampler_kwargs.setdefault("num_warmup", 1000)
    sampler_kwargs.setdefault("num_samples", 1000)
    sampler_kwargs.setdefault("num_chains", 4)
    fit = sampler.sample(**sampler_kwargs)
    return az.from_pystan(fit)
