from typing import Optional
import warnings

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import numpyro
import numpyro.distributions as dist

from .preprocess import deconvolve
from .scale import Scaler
from .utils import outlier_frac, sigma_68


def xdgmm_prior(
    x_scaled: ArrayLike,
    cov_x_scaled: ArrayLike,
    seed: int,
    K: Optional[int] = None,
):
    x_prior = deconvolve(
        x_scaled,
        cov_x_scaled,
        n_components=K,
        random_state=seed,
    )

    if x_prior["weights"].shape[0] == 1:
        return dist.MultivariateNormal(
            loc=x_prior["means"][0],
            covariance_matrix=x_prior["vars"][0],
        )
    else:
        mixing_dist = dist.CategoricalProbs(jnp.array(x_prior["weights"]))

        components = []
        for mu, var in zip(x_prior["means"], x_prior["vars"]):
            components.append(
                dist.MultivariateNormal(
                    loc=mu,
                    covariance_matrix=var,
                )
            )

        return dist.MixtureGeneral(mixing_dist, components)


def model_builder(true_x_prior):
    def tcup_model(
        x_scaled,
        y_scaled,
        cov_x_scaled,
        dy_scaled,
        nu=None,
        nu_prior=None,
        sigma_prior=None,
    ):
        # Prior on heavy-tailedness
        if nu is None:
            if nu_prior is None:
                nu = numpyro.sample("nu", dist.InverseGamma(3, 10))
            else:
                nu = numpyro.sample("nu", nu_prior)

        x_true = numpyro.sample(
            "x_true", true_x_prior, sample_shape=(x_scaled.shape[0],)
        )

        # Priors on regression parameters
        alpha = numpyro.sample("alpha_scaled", dist.Normal(0, 3))
        beta = numpyro.sample(
            "beta", dist.Cauchy(0, 1), sample_shape=(x_scaled.shape[1],)
        )
        sigma = numpyro.sample("sigma_scaled", dist.Gamma(2, 2))

        # Linear regression model
        epsilon = numpyro.sample(
            "epsilon", dist.StudentT(nu, 0, sigma), sample_shape=y_scaled.shape
        )
        y_true = numpyro.deterministic(
            "y_true", alpha + jnp.dot(x_true, beta) + epsilon
        )

        # Measure latent x and y values with error
        numpyro.sample(
            "x_scaled",
            dist.MultivariateStudentT(nu, x_true, cov_x_scaled),
            obs=x_scaled,
        )
        numpyro.sample(
            "y_scaled", dist.StudentT(nu, y_true, dy_scaled), obs=y_scaled
        )

    return tcup_model


def tcup(
    x: ArrayLike,
    y: ArrayLike,
    dy: ArrayLike,
    dx: Optional[ArrayLike] = None,
    cov_x: Optional[ArrayLike] = None,
    model: str = "tcup",
    shape_param: Optional[float] = None,
    seed: Optional[int] = None,
    prior_samples: Optional[int] = 1000,
    model_kwargs: Optional[dict] = None,
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

    if model_kwargs is None:
        model_kwargs = {}

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

    x_true_prior = xdgmm_prior(scaled_x, scaled_cov_x, seed)
    tcup_model = model_builder(x_true_prior)

    # Setup random key
    if seed is None:
        seed = np.random.randint(4294967296)
    rng_key = jax.random.PRNGKey(seed)

    # Set up NUTS kernel
    kernel = numpyro.infer.NUTS(tcup_model)

    # Set up sampler
    sampler_kwargs.setdefault("num_warmup", 1000)
    sampler_kwargs.setdefault("num_samples", 1000)
    sampler_kwargs.setdefault("num_chains", 4)
    sampler_kwargs.setdefault("chain_method", "parallel")
    mcmc = numpyro.infer.MCMC(kernel, **sampler_kwargs)

    # Sample model
    rng_key, rng_key_ = jax.random.split(rng_key)
    mcmc.run(
        rng_key_,
        x_scaled=scaled_x,
        y_scaled=scaled_y,
        cov_x_scaled=scaled_cov_x,
        dy_scaled=scaled_dy,
        extra_fields=["num_steps", "energy"],
    )
    samples = mcmc.get_samples()

    # Sample posterior predictive
    rng_key, rng_key_ = jax.random.split(rng_key)
    post_pred = numpyro.infer.Predictive(tcup_model, samples)(
        rng_key_,
        x_scaled=scaled_x,
        y_scaled=scaled_y,
        cov_x_scaled=scaled_cov_x,
        dy_scaled=scaled_dy,
    )

    # Sample prior
    rng_key, rng_key_ = jax.random.split(rng_key)
    prior = numpyro.infer.Predictive(tcup_model, num_samples=prior_samples)(
        rng_key_,
        x_scaled=scaled_x,
        y_scaled=scaled_y,
        cov_x_scaled=scaled_cov_x,
        dy_scaled=scaled_dy,
    )

    # Combine results into ArviZ InferenceData object
    results = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=post_pred,
    )

    return results
