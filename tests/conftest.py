import numpy as np
import pytest
import scipy.stats as sps

# Set up run parameters
SEED = 24601
N = 12
X_RANGE = (0, 10)
OBS_ERR = 0.2
OUTLIER = 10
PARAMS = {"alpha": 3, "beta": [2], "sigma_int": 0.1}


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

    # Calculate correlation matrices
    rho = np.array([np.diag(np.ones(x_i.shape)) for x_i in x_obs])

    # Return data dictionaries
    outlier_data = {
        "x": x_obs,
        "cov_x": x_cov,
        "y": y_obs,
        "dy": y_err,
        # "rho": rho.tolist(),
    }
    masked_data = {
        "N": N - 1,
        "K": x_true.shape[1],
        "x": x_obs[outlier_mask].tolist(),
        "cov_x": x_cov[outlier_mask].tolist(),
        "y": y_obs[outlier_mask].tolist(),
        "dy": y_err[outlier_mask].tolist(),
        "rho": rho[outlier_mask].tolist(),
    }
    return outlier_data, masked_data


@pytest.fixture
def outlier_data():
    # Set up random number generator
    rng = np.random.default_rng(SEED)

    # Create data
    x_true = np.linspace(*X_RANGE, N)[:, np.newaxis]
    outlier_data, _ = gen_data(
        rng,
        x_true=x_true,
        dx=OBS_ERR,
        dy=OBS_ERR,
        outlier=OUTLIER,
        **PARAMS,
    )

    return outlier_data
