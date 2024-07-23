"""Routines for pre-processing the data before fitting."""

from xdgmm import XDGMM

from .utils import suppress_output


def deconvolve(x, cov_x, n_components=None, random_state=None):
    xdgmm = XDGMM(random_state=random_state)

    if n_components is None:
        with suppress_output():
            _, optimal_n_comp, _ = xdgmm.bic_test(x, cov_x, range(1, 10))
        xdgmm.n_components = optimal_n_comp
    else:
        xdgmm.n_components = n_components

    xdgmm = xdgmm.fit(x, cov_x)

    return {
        "weights": xdgmm.weights,
        "means": xdgmm.mu,
        "vars": xdgmm.V,
    }
