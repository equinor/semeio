import os
import shutil

import pytest
from ert import LibresFacade


@pytest.fixture()
def snake_oil_facade(
    copy_snake_oil_case_storage,
):  # pylint: disable=unused-argument
    yield LibresFacade.from_config_file("snake_oil.ert")


@pytest.fixture()
def setup_poly_ert(tmpdir, test_data_root):
    cwd = os.getcwd()
    tmpdir.chdir()
    test_data_dir = os.path.join(test_data_root, "poly_normal")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    yield
    os.chdir(cwd)
