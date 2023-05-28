import importlib.resources as pkg_resources
from typing import Optional
import warnings

import arviz as az
import numpy as np
from numpy.typing import ArrayLike
import stan

from . import stan_models
from .preprocess import deconvolve
from .scale import Scaler
from .utils import outlier_frac, sigma_68


def _get_model_src(model):
    try:
        return pkg_resources.read_text(stan_models, f"{model}.stan")
    except FileNotFoundError:
        raise NotImplementedError(f"The model `{model}` could not be found.")


def _prep_data(
    x_scaled: ArrayLike,
    cov_x_scaled: ArrayLike,
    y_scaled: ArrayLike,
    dy_scaled: ArrayLike,
    seed: int,
    K: Optional[int],
):
    x_prior = deconvolve(
        x_scaled,
        cov_x_scaled,
        n_components=K,
        random_state=seed,
    )

    (N, D) = x_scaled.shape
    stan_data = {
        "N": N,
        "D": D,
        "x": x_scaled.tolist(),
        "cov_x": cov_x_scaled.tolist(),
        "y": y_scaled.tolist(),
        "dy": dy_scaled.tolist(),
        "K": x_prior["weights"].shape[0],
        "theta_mix": x_prior["weights"].tolist(),
        "mu_mix": x_prior["means"].tolist(),
        "sigma_mix": x_prior["vars"].tolist(),
    }

    return stan_data


def _add_to_fit(
    fit: stan.fit.Fit,
    var_name: str,
    data: ArrayLike,
):
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


def _reprocess_samples(
    scaler: Scaler,
    fit: stan.fit.Fit,
):
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

    # Reshape arrays
    alpha.shape = (1, draws, chains)
    beta.shape = (dim_x, draws, chains)
    sigma.shape = (1, draws, chains)

    _add_to_fit(fit, "alpha_rescaled", alpha)
    _add_to_fit(fit, "beta_rescaled", beta)
    _add_to_fit(fit, "sigma_rescaled", sigma)

    if "nu" in fit.param_names:
        nu_idx = fit._parameter_indexes("nu")
        nu = fit._draws[nu_idx, :, :]
        _add_to_fit(
            fit,
            "sigma_68",
            sigma_68(nu) * sigma,
        )
        _add_to_fit(
            fit,
            "outlier_frac",
            outlier_frac(nu),
        )

    return fit


def tcup(
    x: ArrayLike,
    y: ArrayLike,
    dy: ArrayLike,
    dx: Optional[ArrayLike] = None,
    cov_x: Optional[ArrayLike] = None,
    model: str = "tcup",
    shape_param: Optional[float] = None,
    seed: Optional[int] = None,
    **sampler_kwargs,
):
    if model not in ["tcup", "ncup", "fixed"]:
        raise NotImplementedError(
            "Please choose a model from ['tcup', 'ncup', 'fixed']"
        )

    if model == "fixed":
        if shape_param is None:
            raise ValueError(
                "Shape parameter must be specified for fixed model"
            )
        elif shape_param <= 0:
            raise ValueError("Shape parameter must be a positive real number")
    else:
        if shape_param is not None:
            warnings.warn(f"`shape_param` is ignored for {model} model")

    if dx is not None:
        match dx.shape:
            case (N, D1, D2):
                warnings.warn(
                    "`dx` appears to be an array of covariance matrices (and"
                    "is assumed to be such); to silence this warning, pass as"
                    "`cov_x` instead.",
                    UserWarning,
                )
                if D1 != D2:
                    raise ValueError("Covariance matrices are not square")
                cov_x = dx
            case (N, D):
                cov_x = (
                    np.array([np.identity(D) for _ in range(N)])
                    * dx[:, :, np.newaxis]
                    * dx[:, np.newaxis, :]
                )
            case (N,):
                cov_x = np.ones((N, 1, 1)) * dx.reshape(N, 1, 1) ** 2

    if cov_x is None:
        raise ValueError(
            "Couldn't identify x error data;"
            "please pass either `dx` or `cov_x`"
        )

    if x.ndim == 1:
        x = x[:, np.newaxis]
    scaler = Scaler(x, cov_x, y, dy)

    (
        scaled_x,
        scaled_cov_x,
        scaled_y,
        scaled_dy,
    ) = scaler.transform(x, cov_x, y, dy)

    stan_data = _prep_data(scaled_x, scaled_cov_x, scaled_y, scaled_dy, seed)

    if shape_param is not None:
        stan_data["shape_param"] = shape_param

    model_src = _get_model_src(model)

    sampler = stan.build(model_src, stan_data, random_seed=seed)

    sampler_kwargs.setdefault("num_warmup", 1000)
    sampler_kwargs.setdefault("num_samples", 1000)
    sampler_kwargs.setdefault("num_chains", 4)
    fit = sampler.sample(**sampler_kwargs)
    return az.from_pystan(_reprocess_samples(scaler, fit))
