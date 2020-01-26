import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--runslurm",
        action="store_true",
        default=False,
        help="run slurm tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow to run"
    )
    config.addinivalue_line(
        "markers", "slurm: mark test which require slurm to run"
    )


def pytest_collection_modifyitems(config, items):
    def mark_skip(keyword):
        if config.getoption(f"--run{keyword}"):
            return
        skip = pytest.mark.skip(reason=f"need --run{keyword} option to run")
        for item in items:
            if keyword in item.keywords:
                item.add_marker(skip)

    mark_skip("slow")
    mark_skip("slurm")
