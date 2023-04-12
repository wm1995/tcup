import numpy as np
import pytest
import scipy.stats as sps


def gen_data(rng, alpha, beta, sigma_int, x_true, dx, dy, outlier):
    N = x_true.shape[0]

    # Set up outlier mask
    outlier_mask = np.ones(N, dtype=bool)
    outlier_mask[-2] = False

    # Generate true y values
    y_true = (
        alpha
        + np.dot(x_true, beta)
        + sps.norm.rvs(0, sigma_int, size=(N,), random_state=rng)
    )
    y_true[~outlier_mask] -= outlier

    # Generate observed values
    x_obs = x_true + sps.norm.rvs(0, dx, size=x_true.shape, random_state=rng)
    x_cov = np.array([np.identity(x_i.shape[0]) for x_i in x_obs]) * dx**2
    y_obs = y_true + sps.norm.rvs(0, dy, size=(N,), random_state=rng)
    y_err = np.ones(N) * dy

    # Return data dictionaries
    outlier_data = {
        "x": x_obs,
        "cov_x": x_cov,
        "y": y_obs,
        "dy": y_err,
    }
    return outlier_data


@pytest.fixture(
    params=[
        {
            "seed": 24601,
            "N": 12,
            "x_range": (0, 10),
            "obs_err": 0.2,
            "outlier": 10,
            "params": {"alpha": 3, "beta": [2], "sigma_int": 0.1},
        }
    ]
)
def outlier_data(request):
    # Set up random number generator
    rng = np.random.default_rng(request.param["seed"])

    # Create data
    x_true = np.linspace(*request.param["x_range"], request.param["N"])[
        :, np.newaxis
    ]
    outlier_data = gen_data(
        rng,
        x_true=x_true,
        dx=request.param["obs_err"],
        dy=request.param["obs_err"],
        outlier=request.param["outlier"],
        **request.param["params"],
    )

    return outlier_data
