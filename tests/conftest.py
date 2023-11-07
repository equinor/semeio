import fileinput
import os
import shutil
import subprocess

import cwrap
import pytest
from resdata import ResDataType
from resdata.grid import GridGenerator
from resdata.resfile import ResdataKW

from tests import legacy_test_data


@pytest.fixture(name="test_data_root")
def fixture_test_data_root():
    yield legacy_test_data.__path__[0]


@pytest.fixture()
def setup_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield


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


def pytest_addoption(parser):
    parser.addoption(
        "--ert-integration",
        action="store_true",
        default=False,
        help="Run ERT integration tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ert-integration"):
        # Do not skip tests when --ert-integration is supplied on pytest command line
        return
    skip_ert_integration = pytest.mark.skip(
        reason="need --ert-integration option to run"
    )
    for item in items:
        if "ert-integration" in item.keywords:
            item.add_marker(skip_ert_integration)


@pytest.fixture
def copy_snake_oil_case_storage(_shared_snake_oil_case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(_shared_snake_oil_case, "test_data")
    monkeypatch.chdir("test_data")


def _run_snake_oil(source_root, grid_prop):
    shutil.copytree(os.path.join(source_root, "snake_oil"), "test_data")
    os.chdir("test_data")
    os.makedirs("fields")
    grid = GridGenerator.create_rectangular((10, 12, 5), (1, 1, 1))
    for iens in range(10):
        grid_prop("PERMX", 10, grid.getGlobalSize(), f"fields/permx{iens}.grdecl")
        grid_prop("PORO", 0.2, grid.getGlobalSize(), f"fields/poro{iens}.grdecl")

    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line_nr, line in enumerate(fin):
            if line_nr == 1:
                print("QUEUE_OPTION LOCAL MAX_RUNNING 5", end="")
            if "NUM_REALIZATIONS 25" in line:
                print("NUM_REALIZATIONS 5", end="")
            else:
                print(line, end="")
        subprocess.call(["ert", "ensemble_experiment", "snake_oil.ert"])


@pytest.fixture
def _shared_snake_oil_case(request, monkeypatch, test_data_root, grid_prop):
    """This fixture will run the snake_oil case to populate storage,
    this is quite slow, but the results will be cached. If something comes
    out of sync, clear the cache and start again.
    """
    snake_path = request.config.cache.mkdir(
        "snake_oil_data" + os.environ.get("PYTEST_XDIST_WORKER", "")
    )
    monkeypatch.chdir(snake_path)
    if not os.listdir(snake_path):
        _run_snake_oil(test_data_root, grid_prop)
    else:
        monkeypatch.chdir("test_data")

    yield os.getcwd()


@pytest.fixture(name="grid_prop")
def fixture_grid_prop():
    def wrapper(prop_name, value, grid_size, fname):
        prop = ResdataKW(prop_name, grid_size, ResDataType.RD_FLOAT)
        prop.assign(value)
        with cwrap.open(fname, "w") as file:
            prop.write_grdecl(file)

    return wrapper
