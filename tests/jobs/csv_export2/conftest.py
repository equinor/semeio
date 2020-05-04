import os
import shutil

import pytest

TEST_DATA_DIR = os.environ.get("ERT_TESTDATA_ROOT")
TEST_DATA_DIR_BERGEN = "/d/proj/bg/enkf/ErtTestData"
TEST_DATA_DIR_STAVANGER = "/project/res-testdata/ErtTestData"


def find_available_test_data():
    if TEST_DATA_DIR is None:
        if os.path.isdir(TEST_DATA_DIR_STAVANGER):
            return TEST_DATA_DIR_STAVANGER
        elif os.path.isdir(TEST_DATA_DIR_BERGEN):
            return TEST_DATA_DIR_BERGEN

    return TEST_DATA_DIR


def has_equinor_test_data():
    return os.path.isdir(find_available_test_data())


def pytest_runtest_setup(item):
    if item.get_closest_marker("equinor_test") and not has_equinor_test_data():
        pytest.skip("Test requires Equinor data")


@pytest.fixture()
def ert_statoil_test_data(tmpdir):
    cwd = os.getcwd()
    tmpdir.chdir()
    test_data_dir = os.path.join(find_available_test_data(), "ert-statoil", "spotfire")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    yield

    os.chdir(cwd)


@pytest.fixture()
def setup_tmpdir(tmpdir):
    cwd = os.getcwd()
    tmpdir.chdir()
    yield
    os.chdir(cwd)
