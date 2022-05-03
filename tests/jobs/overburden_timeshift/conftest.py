import os
import shutil

import pytest

from tests import test_data

TEST_NORNE_DIR = os.path.realpath(os.path.join(test_data.__path__[0], "norne"))


@pytest.fixture(scope="session")
def _setup_ots_tmppath(tmpdir_factory):
    # for the sake of python2.7 the path needs to be converted to string
    tmp_path = tmpdir_factory.mktemp("norne", numbered=False).join("test-data").strpath
    shutil.copytree(TEST_NORNE_DIR, tmp_path)
    return tmp_path


# use this fixture for setting up norne data and entering
# another folder while having handle to norne data dir
@pytest.fixture
def ots_tmpdir_enter(_setup_ots_tmppath, tmpdir):
    tmp_path = _setup_ots_tmppath
    with tmpdir.as_cwd():
        yield tmp_path
