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
