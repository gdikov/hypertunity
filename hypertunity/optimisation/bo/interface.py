# -*- coding: utf-8 -*-


__all__ = [
    "BayesianOptimisation",
    "BayesianOptimization"
]

_AVAILABLE_BACKENDS = {"gpyopt"}


def _get_backend_by_name(backend, *args, **kwargs):
    """Get the backend by given name.

    To prevent the interface module from importing all backends, which in turn import other external libraries,
    the backend is imported only after requesting it.

    Args:
        backend: str with the name of the requested backend.

    Returns:
        The class of the BO backend.
    """
    normalised_name = backend.lower().strip()
    if normalised_name in _AVAILABLE_BACKENDS:
        from ._backends import GPyOptBackend
        return GPyOptBackend(*args, **kwargs)
    else:
        raise ValueError(f"Unknown backend {backend}. Currently only {_AVAILABLE_BACKENDS} are supported.")


BayesianOptimisation = _get_backend_by_name
BayesianOptimization = BayesianOptimisation     # alias for the American spelling
