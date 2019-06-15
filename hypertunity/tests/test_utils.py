from .. import utils


def test_support_american_spelling():
    @utils.support_american_spelling
    def gb_spelling_func(minimise, optimise, maximise):
        return minimise, optimise, maximise

    assert gb_spelling_func(minimise=True, optimise=1, maximise=None) == (True, 1, None)
    assert gb_spelling_func(minimize=True, optimize=1, maximize=None) == (True, 1, None)
