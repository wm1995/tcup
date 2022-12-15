import importlib.resources as pkg_resources
import arviz as az
import numpy as np
import stan

from . import stan_models


def prep_data(data):
    # Extract data shape
    match data["x_obs"].shape:
        case (N, K):
            stan_data = {
                "N": N,
                "K": K,
                "x": data["x_obs"].tolist(),
                "dx": data["dx_obs"].tolist(),
                "y": data["y_obs"].tolist(),
                "dy": data["dy_obs"].tolist(),
                "rho": [np.diag(np.ones(x_i.shape)) for x_i in data["x_obs"]],
            }
        case (N,):
            stan_data = {
                "N": N,
                "K": 1,
                "x": data["x_obs"][:, np.newaxis].tolist(),
                "dx": data["dx_obs"][:, np.newaxis].tolist(),
                "y": data["y_obs"].tolist(),
                "dy": data["dy_obs"].tolist(),
                "rho": np.ones((N, 1, 1)),
            }

    return stan_data


def prep_tcup_data(data):
    stan_data = prep_data(data)

    # If nu is not provided, set to -1 to infer as part of model
    stan_data["shape_param"] = data.get("nu", -1)

    return stan_data


def tcup(data, seed=None, **sampler_kwargs):
    stan_data = prep_tcup_data(data)

    model_src = pkg_resources.read_text(stan_models, "tcup.stan")

    sampler = stan.build(model_src, stan_data, random_seed=seed)

    sampler_kwargs.setdefault("num_warmup", 1000)
    sampler_kwargs.setdefault("num_samples", 1000)
    sampler_kwargs.setdefault("num_chains", 4)
    fit = sampler.sample(**sampler_kwargs)
    return az.from_pystan(fit)


def ncup(data, seed=None, **sampler_kwargs):
    ncup_data = prep_data(data)

    model = pkg_resources.read_text(stan_models, "ncup.stan")

    sampler = stan.build(model, ncup_data, random_seed=seed)

    sampler_kwargs.setdefault("num_warmup", 1000)
    sampler_kwargs.setdefault("num_samples", 1000)
    sampler_kwargs.setdefault("num_chains", 4)
    fit = sampler.sample(**sampler_kwargs)

    return az.from_pystan(fit)
