from typing import Optional

from numpy.typing import ArrayLike

from .numpyro import tcup as _tcup_numpyro
from .stan import tcup as _tcup_stan


def tcup(
    x: ArrayLike,
    y: ArrayLike,
    dy: ArrayLike,
    dx: Optional[ArrayLike] = None,
    cov_x: Optional[ArrayLike] = None,
    seed: Optional[int] = None,
    backend: Optional[str] = "stan",
    **backend_kwargs,
):
    match backend:
        case "numpyro":
            return _tcup_numpyro(
                x, y, dy, dx, cov_x, seed=seed, **backend_kwargs
            )
        case "stan":
            return _tcup_stan(x, y, dy, dx, cov_x, seed=seed, **backend_kwargs)
        case _:
            raise NotImplementedError(f"Backend {backend} not recognised")
