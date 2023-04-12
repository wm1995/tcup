import importlib.resources as pkg_resources
import arviz as az
import numpy as np
import stan

from . import stan_models
from .preprocess import deconvolve
from .scale import Scaler


def _get_model_src(model):
    try:
        return pkg_resources.read_text(stan_models, f"{model}.stan")
    except FileNotFoundError:
        raise NotImplementedError(f"The model `{model}` could not be found.")


def _prep_data(data, seed):
    # If nu is not provided, set to -1 to infer as part of model
    shape_param = data.get("nu", -1)

    x_prior = deconvolve(
        data["x"],
        data["dx"],
        n_components=data.get("K"),
        random_state=seed,
    )

    stan_data = {
        "y": data["y"].tolist(),
        "dy": data["dy"].tolist(),
        "shape_param": shape_param,
        "K": x_prior["weights"].shape[0],
        "theta_mix": x_prior["weights"].tolist(),
        "mu_mix": x_prior["means"].tolist(),
        "sigma_mix": x_prior["vars"].tolist(),
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


def _add_to_fit(fit, var_name, data):
    D = data.shape[0]

    fit.param_names += (var_name,)

    if D == 1:
        fit.dims += ([],)
        fit.constrained_param_names += (var_name,)
    else:
        fit.dims += ([D],)
        fit.constrained_param_names += tuple(
            f"{var_name}.{i}" for i in range(D)
        )

    fit._draws = np.append(fit._draws, data, axis=0)


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

    _add_to_fit(fit, "alpha_rescaled", alpha.reshape(1, draws, chains))
    _add_to_fit(fit, "beta_rescaled", beta.reshape(dim_x, draws, chains))
    _add_to_fit(fit, "sigma_rescaled", sigma.reshape(1, draws, chains))

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

    stan_data = _prep_data(scaled_data, seed)

    model_src = _get_model_src(model)

    sampler = stan.build(model_src, stan_data, random_seed=seed)

    sampler_kwargs.setdefault("num_warmup", 1000)
    sampler_kwargs.setdefault("num_samples", 1000)
    sampler_kwargs.setdefault("num_chains", 4)
    fit = sampler.sample(**sampler_kwargs)
    return az.from_pystan(_reprocess_samples(scaler, fit))
