import copy
import os
import shutil
from pathlib import Path

import pytest

ECLBASE_NORNE: str = str(
    Path(__file__).resolve().parent / "data" / "norne" / "NORNE_ATW2013"
)
ECLBASE_REEK: str = str(
    Path(__file__).resolve().parent / "data" / "reek" / "2_R001_REEK-0"
)
EXPECTED_RESULTS_PATH_NORNE: Path = (
    Path(__file__).resolve().parent / "data" / "norne" / "expected_result"
)

MOCK_DATA_CONTENT_NORNE = {
    "RFT_B-1AH_0_active": [1] * 5,
    "RFT_B-4AH_0_active": [1] * 29,
    "RFT_B-4H_0_active": [1] * 20,
    "RFT_C-1H_0_active": [1] * 21,
    "RFT_C-3H_0_active": [1] * 21,
    "RFT_C-4AH_0_active": [1] * 15,
}


def get_ecl_base_norne():
    return copy.copy(ECLBASE_NORNE)


def get_ecl_base_reek():
    return copy.copy(ECLBASE_REEK)


def get_expected_results_path_norne():
    return copy.copy(EXPECTED_RESULTS_PATH_NORNE)


def get_mock_data_content_norne():
    return copy.deepcopy(MOCK_DATA_CONTENT_NORNE)


def _generate_mock_data_norne(write_directory: Path):
    for fname, content in MOCK_DATA_CONTENT_NORNE.items():
        with (write_directory / fname).open("w+", encoding="utf-8") as file:
            file.write("\n".join([str(c) for c in content]))


@pytest.fixture
def norne_data(tmpdir):
    data_dir = (
        Path(__file__).resolve().parent / "data" / "norne" / "gendata_rft_input_files"
    )
    shutil.copytree(data_dir, tmpdir.strpath, dirs_exist_ok=True)

    with tmpdir.as_cwd():
        expected_results_dir = Path(tmpdir.strpath) / "expected_results"
        expected_results_dir.mkdir()

        shutil.copytree(
            EXPECTED_RESULTS_PATH_NORNE, expected_results_dir, dirs_exist_ok=True
        )
        _generate_mock_data_norne(expected_results_dir)
        yield


@pytest.fixture
def reek_data(tmpdir):
    data_dir = (
        Path(__file__).resolve().parent / "data" / "reek" / "gendata_rft_input_files"
    )
    shutil.copytree(data_dir, tmpdir.strpath, dirs_exist_ok=True)
    cwd = Path.cwd()
    tmpdir.chdir()

    try:
        yield

    finally:
        os.chdir(cwd)
