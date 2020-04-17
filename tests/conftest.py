import os
import pytest


@pytest.fixture()
def setup_tmpdir(tmpdir):
    cwd = os.getcwd()
    tmpdir.chdir()
    try:
        yield
    finally:
        os.chdir(cwd)
