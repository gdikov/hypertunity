

GB_US_STYLE = {"minimise": "minimize",
               "maximise": "maximize",
               "optimise": "optimize"}

US_GB_STYLE = {us: gb for gb, us in GB_US_STYLE.items()}


def support_american_style(func_or_class):
    """Convert American-style keyword arguments to British (default for hypertunity).

    Args:
        func_or_class: a Python callable to decorate.

    Returns:
        The decorated function which supports American keyword arguments.
    """
    def british_style_func(*args, **kwargs):
        gb_kwargs = {US_GB_STYLE.get(kw, kw): val for kw, val in kwargs.items()}
        return func_or_class(*args, **gb_kwargs)

    return british_style_func
