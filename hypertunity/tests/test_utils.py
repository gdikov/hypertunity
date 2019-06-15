from .. import utils


def test_support_american_style():
    @utils.support_american_style
    def gb_style_func(minimise, optimise, maximise):
        return minimise, optimise, maximise

    assert gb_style_func(minimise=True, optimise=1, maximise=None) == (True, 1, None)
    assert gb_style_func(minimize=True, optimize=1, maximize=None) == (True, 1, None)
