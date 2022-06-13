import os
import gc
import pytest
from tests import legacy_test_data


@pytest.fixture()
def test_data_root():
    yield legacy_test_data.__path__[0]


@pytest.fixture()
def setup_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield
    # This makes sure EnkFMain is cleaned up between
    # each test, otherwise a storage folder is created
    # in semeio root dir at the end of the test run.
    # Should be deleted when we no longer use EnkFMain
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def move_to_shared_tmp(tmp_path_factory):
    """
    We do this because when running some of the tests individually
    they create a storage folder in the semeio root directory during
    tear down. This places that storage folder in a shared tempdir
    instead. Should be deleted when we no longer use EnkFMain
    """
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("some_shared_path"))
    yield
    os.chdir(cwd)


def pytest_addoption(parser):
    parser.addoption(
        "--ert_integration",
        action="store_true",
        default=False,
        help="Run ERT integration tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ert_integration"):
        # Do not skip tests when --ert_integration is supplied on pytest command line
        return
    skip_ert_integration = pytest.mark.skip(
        reason="need --ert_integration option to run"
    )
    for item in items:
        if "ert_integration" in item.keywords:
            item.add_marker(skip_ert_integration)
