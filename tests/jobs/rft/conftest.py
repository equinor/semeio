import copy
import os
import shutil

import pytest

ECL_BASE_NORNE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/norne/NORNE_ATW2013"
)
ECL_BASE_REEK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/reek/2_R001_REEK-0"
)
EXPECTED_RESULTS_PATH_NORNE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/norne/expected_result"
)

MOCK_DATA_CONTENT_NORNE = {
    "RFT_B-1AH_0_active": [1] * 5,
    "RFT_B-4AH_0_active": [1] * 7 + [0] * 1 + [1] * 21,
    "RFT_B-4H_0_active": [1] * 20,
    "RFT_C-1H_0_active": [1] * 21,
    "RFT_C-3H_0_active": [1] * 21,
    "RFT_C-4AH_0_active": [1] * 15,
}


def get_ecl_base_norne():
    return copy.copy(ECL_BASE_NORNE)


def get_ecl_base_reek():
    return copy.copy(ECL_BASE_REEK)


def get_expected_results_path_norne():
    return copy.copy(EXPECTED_RESULTS_PATH_NORNE)


def get_mock_data_content_norne():
    return copy.deepcopy(MOCK_DATA_CONTENT_NORNE)


def _generate_mock_data_norne(write_directory):
    for fname, content in MOCK_DATA_CONTENT_NORNE.items():
        with open(os.path.join(write_directory, fname), "w+", encoding="utf-8") as file:
            file.write("\n".join([str(c) for c in content]))


@pytest.fixture
def norne_data(tmpdir):
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/norne/gendata_rft_input_files"
    )
    shutil.copytree(data_dir, tmpdir.strpath, dirs_exist_ok=True)

    cwd = os.getcwd()
    tmpdir.chdir()

    expected_results_dir = os.path.join(tmpdir.strpath, "expected_results")
    os.mkdir(expected_results_dir)

    shutil.copytree(
        EXPECTED_RESULTS_PATH_NORNE, expected_results_dir, dirs_exist_ok=True
    )
    _generate_mock_data_norne(expected_results_dir)

    try:
        yield

    finally:
        os.chdir(cwd)


@pytest.fixture
def reek_data(tmpdir):
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/reek/gendata_rft_input_files"
    )
    shutil.copytree(data_dir, tmpdir.strpath, dirs_exist_ok=True)
    cwd = os.getcwd()
    tmpdir.chdir()

    try:
        yield

    finally:
        os.chdir(cwd)
