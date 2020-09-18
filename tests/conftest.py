import os
import pytest
from tests import legacy_test_data


@pytest.fixture()
def test_data_root():
    yield legacy_test_data.__path__[0]


@pytest.fixture()
def setup_tmpdir(tmpdir):
    cwd = os.getcwd()
    tmpdir.chdir()
    yield
    os.chdir(cwd)
