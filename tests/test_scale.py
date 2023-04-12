import numpy as np
import pytest
from tcup.scale import Scaler

ALPHA = 2
BETA = [1, -3]
SIGMA = 0.5


@pytest.fixture
def simple_linear_data():
    N = 5
    dim_x = len(BETA)
    x = np.vstack([np.logspace(0, 1, N), np.linspace(-1, -4, N)]).T
    y = ALPHA + np.dot(x, BETA)

    dx = np.identity(dim_x)
    if dim_x > 1:
        for i in range(1, dim_x):
            dx[i, i - 1] = 0.5
            dx[i - 1, i] = 0.5

    data = {
        "x": np.vstack([x, x]),
        "dx": np.broadcast_to(dx, (2 * N, dim_x, dim_x)),
        "y": np.hstack([y - SIGMA, y + SIGMA]),
        "dy": np.ones(2 * N),
    }
    return data


def test_data_fixture(simple_linear_data):
    x = simple_linear_data["x"]
    y = simple_linear_data["y"]
    A = np.append(x, np.ones(x.shape[0])[:, np.newaxis], axis=1)
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    beta = coeffs[:-1]
    alpha = coeffs[-1]
    residuals = y - A @ coeffs
    assert np.isclose(alpha, ALPHA)
    assert np.isclose(beta, BETA).all()
    assert np.isclose(residuals.std(), SIGMA)


def test_scaler(simple_linear_data):
    x = simple_linear_data["x"]
    dim_x = x.shape[1]
    dx = simple_linear_data["dx"]
    y = simple_linear_data["y"]
    dy = simple_linear_data["dy"]
    scaler = Scaler(**simple_linear_data)
    x_scaled, dx_scaled, y_scaled, dy_scaled = scaler.transform(
        **simple_linear_data
    )

    # Check means are 0 and variances are 1
    assert np.isclose(x_scaled.mean(), 0).all()
    assert np.isclose(x_scaled.std(), 1).all()
    assert np.isclose(y_scaled.mean(), 0)
    assert np.isclose(y_scaled.std(), 1)

    std_x = x.std(axis=0)

    # Check diagonal elements of covariance matrix are scaled correctly
    assert np.isclose(np.diag(dx_scaled[0]), np.diag(dx[0]) / std_x**2).all()
    # Check off-diagonal elements are scaled correctly
    if dim_x > 1:
        assert np.isclose(
            np.diag(dx_scaled[0, 1:, :-1]),
            dx[0, 1:, :-1] / std_x[:-1] / std_x[1:],
        ).all()

    std_y = y.std(axis=0)

    # Check dy scaled correctly
    assert np.isclose(dy_scaled, dy / std_y).all()


def test_inv_transform(simple_linear_data):
    scaler = Scaler(**simple_linear_data)
    scaled_data = scaler.transform(**simple_linear_data)
    x, dx, y, dy = scaler.inv_transform(*scaled_data)
    assert np.isclose(x, simple_linear_data["x"]).all()
    assert np.isclose(dx, simple_linear_data["dx"]).all()
    assert np.isclose(y, simple_linear_data["y"]).all()
    assert np.isclose(dy, simple_linear_data["dy"]).all()


def test_transform_coeff(simple_linear_data):
    scaler = Scaler(**simple_linear_data)
    x_scaled, _, y_scaled, _ = scaler.transform(**simple_linear_data)
    A = np.append(x_scaled, np.ones(x_scaled.shape[0])[:, np.newaxis], axis=1)
    coeffs = np.linalg.lstsq(A, y_scaled, rcond=None)[0]
    residuals = y_scaled - A @ coeffs

    # Test forward transform
    alpha_scaled, beta_scaled, sigma_scaled = scaler.transform_coeff(
        ALPHA, BETA, SIGMA
    )
    assert np.isclose(alpha_scaled, coeffs[-1])
    assert np.isclose(beta_scaled, coeffs[:-1]).all()
    assert np.isclose(sigma_scaled, residuals.std())

    # Test backwards transform
    alpha, beta, sigma = scaler.inv_transform_coeff(
        alpha_scaled=coeffs[-1],
        beta_scaled=coeffs[:-1],
        sigma_scaled=residuals.std(),
    )
    assert np.isclose(alpha, ALPHA)
    assert np.isclose(beta, BETA).all()
    assert np.isclose(sigma, SIGMA)
