from .numpyro import tcup as _tcup_numpyro


def tcup(data, seed=None, backend="numpyro", **backend_kwargs):
    match backend:
        case "numpyro":
            return _tcup_numpyro(data, seed, **backend_kwargs)
        case "stan":
            raise NotImplementedError("Stan not yet implemented")
        case _:
            raise NotImplementedError(f"Backend {backend} not recognised")
