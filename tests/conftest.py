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
