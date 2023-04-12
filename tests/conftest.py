import numpy as np
import pytest
import scipy.stats as sps


def gen_data(rng, alpha, beta, sigma_int, x_true, dx, dy, outlier):
    N = x_true.shape[0]
    if x_true.ndim > 1:
        D = x_true.shape[1]

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
    if x_true.ndim > 1:
        x_cov = np.array([np.identity(D) for _ in range(N)]) * dx**2
    else:
        x_err = np.ones(N) * dx
    y_obs = y_true + sps.norm.rvs(0, dy, size=(N,), random_state=rng)
    y_err = np.ones(N) * dy

    # Return data dictionaries
    outlier_data = {
        "x": x_obs,
        "y": y_obs,
        "dy": y_err,
    }

    if x_true.ndim > 1:
        outlier_data["cov_x"] = x_cov
    else:
        outlier_data["dx"] = x_err

    return outlier_data


# def gen_data(rng, alpha, beta, sigma_int, x_true, dx, dy, outlier):
@pytest.fixture(
    params=[
        {
            "rng": np.random.default_rng(24601),
            "alpha": 3,
            "beta": 2,
            "sigma_int": 0.1,
            "x_true": np.linspace(0, 10, 12),
            "dx": 0.2,
            "dy": 0.2,
            "outlier": 10,
        },
        {
            "rng": np.random.default_rng(24601),
            "alpha": 3,
            "beta": [2],
            "sigma_int": 0.1,
            "x_true": np.linspace(0, 10, 12)[:, np.newaxis],
            "dx": 0.2,
            "dy": 0.2,
            "outlier": 10,
        },
        {
            "rng": np.random.default_rng(24601),
            "alpha": -1,
            "beta": [2, 3],
            "sigma_int": 0.3,
            "x_true": np.vstack(
                [
                    np.logspace(0, 1, 20),
                    np.linspace(-1, -4, 20),
                ]
            ).T,
            "dx": 0.2,
            "dy": 0.2,
            "outlier": 10,
        },
    ]
)
def data(request):
    return gen_data(**request.param)
