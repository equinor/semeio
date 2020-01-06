import os
from distutils.dir_util import copy_tree
import numpy
import pytest

from semeio.jobs.scripts.gendata_rft import main_entry_point

ECL_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/norne/NORNE_ATW2013"
)
EXPECTED_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/norne/expected_result"
)

MOCK_DATA_CONTENT = {
    "RFT_B-1AH_0_active": [1] * 5,
    "RFT_B-4AH_0_active": [1] * 7 + [0] * 1 + [1] * 21,
    "RFT_B-4H_0_active": [1] * 20,
    "RFT_C-1H_0_active": [1] * 21,
    "RFT_C-3H_0_active": [1] * 21,
    "RFT_C-4AH_0_active": [1] * 15,
}


def _assert_almost_equal_line_by_line(file1, file2):
    with open(file1) as fh:
        file1_content = fh.readlines()

    with open(file2) as fh:
        file2_content = fh.readlines()

    assert len(file1_content) == len(file2_content)

    for line1, line2 in zip(file1_content, file2_content):
        try:
            line1, line2 = float(line1), float(line2)
        except ValueError:
            continue
        numpy.testing.assert_almost_equal(line1, line2, decimal=7)


def _generate_mock_data(write_directory):
    for fname, content in MOCK_DATA_CONTENT.items():
        with open(os.path.join(write_directory, fname), "w+") as fh:
            fh.write("\n".join([str(c) for c in content]))


@pytest.fixture
def input_data(tmpdir):
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/norne/gendata_rft_input_files"
    )
    copy_tree(data_dir, tmpdir.strpath)

    cwd = os.getcwd()
    tmpdir.chdir()

    expected_results_dir = os.path.join(tmpdir.strpath, "expected_results")
    os.mkdir(expected_results_dir)

    copy_tree(EXPECTED_RESULTS_PATH, expected_results_dir)
    _generate_mock_data(expected_results_dir)

    try:
        yield

    finally:
        os.chdir(cwd)


def test_gendata_rft_entry_point(tmpdir, input_data):

    arguments = [
        "-e",
        ECL_BASE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]

    main_entry_point(arguments)
    expected_results_dir = os.path.join(tmpdir.strpath, "expected_results")

    expected_files = [
        os.path.join(EXPECTED_RESULTS_PATH, fname)
        for fname in os.listdir(EXPECTED_RESULTS_PATH)
    ]

    expected_files.extend(
        [
            os.path.join(expected_results_dir, fname)
            for fname in os.listdir(expected_results_dir)
        ]
    )

    for expected_file in expected_files:
        filename = os.path.basename(expected_file)
        assert filename in os.listdir(tmpdir.strpath)

        if not filename.endswith("inactive_info"):
            result_file = os.path.join(tmpdir.strpath, filename)

            _assert_almost_equal_line_by_line(expected_file, result_file)


def test_gendata_inactive_info_point_not_in_grid(tmpdir, input_data):

    with open("B-1AH.txt", "a+") as fh:
        fh.write("0 1 2 3\n")

    with open("well_and_time.txt", "w+") as fh:
        fh.write("B-1AH 1 12 2005 0\n")

    arguments = [
        "-e",
        ECL_BASE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]
    main_entry_point(arguments)

    with open("RFT_B-1AH_0_inactive_info") as fh:
        result = fh.read()
        assert result.startswith("TRAJECTORY_POINT_NOT_IN_GRID")


def test_gendata_inactive_info_zone_mismatch(tmpdir, input_data):

    with open("well_and_time.txt", "w+") as fh:
        fh.write("B-1AH 1 12 2005 0\n")

    with open(os.path.join(tmpdir.strpath, "B-1AH.txt"), "r+") as fh:
        lines = fh.readlines()

    line = lines[-1].rsplit(" ", 1)[0]
    line = line + " last_zone"

    with open("B-1AH.txt", "w+") as fh:
        fh.write(line)

    arguments = [
        "-e",
        ECL_BASE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]
    main_entry_point(arguments)

    with open("RFT_B-1AH_0_inactive_info") as fh:
        result = fh.read()
        assert result.startswith("ZONE_MISMATCH")


def test_gendata_inactive_info_not_in_rft(tmpdir, input_data):

    with open("well_and_time.txt", "w+") as fh:
        fh.write("B-1AH 1 12 2005 0\n")

    with open(os.path.join(tmpdir.strpath, "B-1AH.txt"), "r+") as fh:
        lines = fh.readlines()

    line = lines[-1].rsplit(" ", 3)[0]
    line += " 2700 2700"

    with open("B-1AH.txt", "w+") as fh:
        fh.write(line)

    arguments = [
        "-e",
        ECL_BASE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]
    main_entry_point(arguments)

    with open("RFT_B-1AH_0_inactive_info") as fh:
        result = fh.read()
        assert result.startswith("TRAJECTORY_POINT_NOT_IN_RFT")


def test_gendata_inactive_info_zone_missing_value(tmpdir, input_data):

    with open("well_and_time.txt", "w+") as fh:
        fh.write("B-1AH 1 12 2005 0\n")

    with open("zonemap.txt", "w+") as fh:
        fh.write("1 zone1\n")

    arguments = [
        "-e",
        ECL_BASE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]

    main_entry_point(arguments)

    with open("RFT_B-1AH_0_inactive_info") as fh:
        result = fh.read()
        assert result.startswith("ZONEMAP_MISSING_VALUE")
