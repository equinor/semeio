import os
from distutils.dir_util import copy_tree
import numpy
import pandas as pd

import pytest

from semeio.jobs.scripts.gendata_rft import main_entry_point, _build_parser

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


def test_gendata_rft_csv(tmpdir, input_data):
    csv_filename = "gendata_rft.csv"
    arguments = [
        "-e",
        ECL_BASE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
        "-c",
        csv_filename,
    ]

    main_entry_point(arguments)
    assert os.path.exists(csv_filename)
    dframe = pd.read_csv(csv_filename)
    assert not dframe.empty

    required_columns = {
        "utm_x",
        "utm_y",
        "measured_depth",
        "true_vertical_depth",
        "zone",
        "i",
        "j",
        "k",
        "pressure",
        "valid_zone",
        "is_active",
        "inactive_info",
        "well",
        "time",
    }
    assert required_columns.issubset(set(dframe.columns))
    assert len(dframe["well"].unique()) == len(MOCK_DATA_CONTENT)

    # i-j-k always non-null together:
    assert (dframe["i"].isnull() == dframe["j"].isnull()).all()
    assert (dframe["i"].isnull() == dframe["k"].isnull()).all()

    # inactive_info must be non-null when active = False
    assert (~dframe[~dframe["is_active"]]["inactive_info"].isnull()).all()


def test_gendata_rft_directory(tmpdir, input_data):
    outputdirectory = "rft_output_dir"
    tmpdir.mkdir(outputdirectory)
    arguments = [
        "-e",
        ECL_BASE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
        "-c",
        "csvfile.csv",
        "-o",
        outputdirectory,
    ]
    main_entry_point(arguments)
    well_count = 6
    files_pr_well = 3
    assert len(list(os.walk(outputdirectory))[0][2]) == well_count * files_pr_well
    assert os.path.exists("csvfile.csv")  # Should be independent of --outputdirectory


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
        assert result.startswith(
            "TRAJECTORY_POINT_NOT_IN_GRID (utm_x=0.0, utm_y=1.0, measured_depth=2.0)"
        )


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
        assert result.startswith("ZONE_MISMATCH (utm_x=")


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
        assert result.startswith("TRAJECTORY_POINT_NOT_IN_RFT (utm_x=")


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
        assert result.startswith("ZONEMAP_MISSING_VALUE (utm_x=")


def test_gendata_partial_rft_file(tmpdir, input_data):
    """Test how the code behaves when some report steps are missing, e.g.
    a Eclipse simulation that has crashed midway.

    We emulate this situation by asking for well-times that are not in the
    binary RFT output in the test dataset.

    In such a situation, we do not want the OK file to written,
    but the CSV file can contain results of data up until the Eclipse crash.
    """

    with open("well_and_time.txt", "a") as file_h:
        # Binary example file does not go as far as 2045:
        file_h.write("B-1AH 1 12 2045 0")

    csv_filename = "gendata_rft.csv"
    arguments = [
        "-e",
        ECL_BASE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
        "-c",
        csv_filename,
    ]

    main_entry_point(arguments)

    assert os.path.exists(csv_filename)
    assert not pd.read_csv(csv_filename).empty
    assert not os.path.exists("GENDATA_RFT.OK")


def test_gendata_rft_empty_well_and_time(tmpdir, input_data):
    def file_count_cwd():
        return len(list(os.walk("."))[0][2])

    with open("empty.txt", "w") as file_h:
        file_h.write("")

    arguments = [
        "-e",
        ECL_BASE,
        "-w",
        "empty.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
        "-c",
        "notwritten.csv",
    ]

    pre_file_count = file_count_cwd()
    main_entry_point(arguments)

    # Empty file is seen as an error, we should not write the OK file, and
    # there should be no CSV file.

    assert not os.path.exists("notwritten.csv")
    assert not os.path.exists("GENDATA_RFT.OK")
    assert file_count_cwd() == pre_file_count


def test_defaults():
    """Default filename for CSV export and outputdir is/can be handled at multiple
    layers.  ERT users infer the default from the JOB_DESCRIPTION file, while command
    line users (or Everest/ERT3?) might get the default from the argparse setup.

    To avoid confusion, these should be in sync, enforced by this test"""

    # Navigate to the JOB_DESCRIPTION in the source code tree:
    import semeio.jobs

    job_description_file = os.path.join(
        os.path.dirname(semeio.jobs.__file__), "config_jobs", "GENDATA_RFT"
    )

    # Crude parsing of the file
    for line in open(job_description_file).readlines():
        if line.startswith("DEFAULT"):
            if line.split()[0:2] == ["DEFAULT", "<CSVFILE>"]:
                csv_job_default = line.split()[2]
            if line.split()[0:2] == ["DEFAULT", "<OUTPUTDIRECTORY>"]:
                directory_job_default = line.split()[2]

    # And compare with argparse:
    assert csv_job_default == _build_parser().get_default("csvfile")
    assert directory_job_default == _build_parser().get_default("outputdirectory")
