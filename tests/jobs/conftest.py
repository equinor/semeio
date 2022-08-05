import os
import shutil

import pytest
from ert import LibresFacade

TEST_DATA_DIR = os.environ.get("LIBRES_TEST_DATA_DIR")


def has_equinor_test_data():
    return os.path.isdir(os.path.join(TEST_DATA_DIR, "Equinor"))


def pytest_runtest_setup(item):
    if item.get_closest_marker("equinor_test") and not has_equinor_test_data():
        pytest.skip("Test requires Equinor data")


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
