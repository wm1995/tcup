import importlib.resources as pkg_resources
import arviz as az
import numpy as np
import stan

from . import stan_models
from .preprocess import Scaler


def _get_model_src(model):
    try:
        return pkg_resources.read_text(stan_models, f"{model}.stan")
    except FileNotFoundError:
        raise NotImplementedError(f"The model `{model}` could not be found.")


def _prep_data(data):
    # If nu is not provided, set to -1 to infer as part of model
    shape_param = data.get("nu", -1)

    stan_data = {
        "y": data["y"].tolist(),
        "dy": data["dy"].tolist(),
        "shape_param": shape_param,
    }

    # Extract data shape
    match data["x"].shape:
        case (N, D):
            stan_data |= {
                "N": N,
                "D": D,
                "x": data["x"].tolist(),
                "dx": data["dx"].tolist(),
            }
        case (N,):
            # Need to reshape x data
            stan_data |= {
                "N": N,
                "D": 1,
                "x": data["x"][:, np.newaxis].tolist(),
                "dx": data["dx"][:, np.newaxis, np.newaxis].tolist(),
            }

    return stan_data


def _reprocess_samples(scaler, fit):
    (_, draws, chains) = fit._draws.shape
    alpha_idx = fit._parameter_indexes("alpha")
    beta_idx = fit._parameter_indexes("beta")
    sigma_idx = fit._parameter_indexes("sigma")
    dim_x = fit.dims[fit.param_names.index("beta")][0]

    alpha_scaled = fit._draws[alpha_idx, :, :].reshape(-1)
    beta_scaled = fit._draws[beta_idx, :, :].reshape(dim_x, -1)
    sigma_scaled = fit._draws[sigma_idx, :, :].reshape(-1)

    alpha, beta, sigma = scaler.inv_transform_coeff(
        alpha_scaled, beta_scaled, sigma_scaled
    )

    fit.param_names += ("alpha_rescaled",)
    fit.constrained_param_names += ("alpha_rescaled",)
    fit.dims += ([],)
    fit._draws = np.append(fit._draws, alpha.reshape(1, draws, chains), axis=0)

    fit.param_names += ("beta_rescaled",)
    fit.constrained_param_names += tuple(
        f"beta_rescaled.{i}" for i in range(dim_x)
    )
    fit.dims += ([dim_x],)
    fit._draws = np.append(
        fit._draws,
        beta.reshape(dim_x, draws, chains),
        axis=0,
    )

    fit.param_names += ("sigma_rescaled",)
    fit.constrained_param_names += ("sigma_rescaled",)
    fit.dims += ([],)
    fit._draws = np.append(fit._draws, sigma.reshape(1, draws, chains), axis=0)

    return fit


def tcup(data, seed=None, model="tcup", **sampler_kwargs):
    x = data.get("x")
    dx = data.get("dx")
    y = data.get("y")
    dy = data.get("dy")

    scaler = Scaler(x, dx, y, dy)

    scaled_data = data.copy()
    (
        scaled_data["x"],
        scaled_data["dx"],
        scaled_data["y"],
        scaled_data["dy"],
    ) = scaler.transform(x, dx, y, dy)

    stan_data = _prep_data(scaled_data)

    model_src = _get_model_src(model)

    sampler = stan.build(model_src, stan_data, random_seed=seed)

    sampler_kwargs.setdefault("num_warmup", 1000)
    sampler_kwargs.setdefault("num_samples", 1000)
    sampler_kwargs.setdefault("num_chains", 4)
    fit = sampler.sample(**sampler_kwargs)
    return az.from_pystan(_reprocess_samples(scaler, fit))
