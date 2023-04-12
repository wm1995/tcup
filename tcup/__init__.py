from .numpyro import tcup as _tcup_numpyro
from .stan import tcup as _tcup_stan


def tcup(data, seed=None, backend="stan", **backend_kwargs):
    match backend:
        case "numpyro":
            return _tcup_numpyro(data, seed, **backend_kwargs)
        case "stan":
            return _tcup_stan(data, seed, **backend_kwargs)
        case _:
            raise NotImplementedError(f"Backend {backend} not recognised")
