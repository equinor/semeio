import os
import shutil

import pytest
from ert import LibresFacade


@pytest.fixture()
def setup_ert(tmpdir, test_data_root):
    cwd = os.getcwd()
    tmpdir.chdir()
    test_data_dir = os.path.join(test_data_root, "snake_oil")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    yield LibresFacade.from_config_file("snake_oil.ert")

    os.chdir(cwd)


@pytest.fixture()
def setup_poly_ert(tmpdir, test_data_root):
    cwd = os.getcwd()
    tmpdir.chdir()
    test_data_dir = os.path.join(test_data_root, "poly_normal")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    yield
    os.chdir(cwd)
