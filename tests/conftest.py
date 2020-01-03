import os
import pytest


@pytest.fixture()
def setup_tmpdir(tmpdir):
    cwd = os.getcwd()
    tmpdir.chdir()
    yield
    os.chdir(cwd)
