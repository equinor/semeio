# pylint: disable=unsubscriptable-object  # pylint issue
import sys
import os
from pathlib import Path

import pandas as pd
import numpy

import pytest

from semeio.jobs.scripts.gendata_rft import main_entry_point, _build_parser
from tests.jobs.rft import conftest

ECL_BASE_NORNE = conftest.get_ecl_base_norne()
ECL_BASE_REEK = conftest.get_ecl_base_reek()
MOCK_DATA_CONTENT_NORNE = conftest.get_mock_data_content_norne()
EXPECTED_RESULTS_PATH_NORNE = conftest.get_expected_results_path_norne()


def test_gendata_rft_csv(tmpdir, norne_data, monkeypatch):
    csv_filename = "gendata_rft.csv"
    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
        "-c",
        csv_filename,
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main_entry_point()
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
    assert len(dframe["well"].unique()) == len(MOCK_DATA_CONTENT_NORNE)

    # i-j-k always non-null together:
    assert (dframe["i"].isnull() == dframe["j"].isnull()).all()
    assert (dframe["i"].isnull() == dframe["k"].isnull()).all()

    # inactive_info must be non-null when active = False
    assert (~dframe[~dframe["is_active"]]["inactive_info"].isnull()).all()


def test_gendata_rft_csv_reek(tmpdir, reek_data, monkeypatch):
    csv_filename = "gendata_rft.csv"
    arguments = [
        "script_name",
        "-e",
        ECL_BASE_REEK,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main_entry_point()
    assert os.path.exists(csv_filename)
    dframe = pd.read_csv(csv_filename)
    assert not dframe.empty
    assert {"pressure", "swat", "sgas"}.issubset(set(dframe.columns))
    assert numpy.isclose(dframe["pressure"].values[0], 304.37)
    assert numpy.isclose(dframe["swat"].values[0], 0.151044)
    assert numpy.isclose(dframe["soil"].values[0], 1 - 0.151044)
    assert numpy.isclose(dframe["sgas"].values[0], 0.0)


def test_gendata_rft_directory(tmpdir, norne_data, monkeypatch):
    outputdirectory = "rft_output_dir"
    tmpdir.mkdir(outputdirectory)
    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
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
    monkeypatch.setattr(sys, "argv", arguments)
    main_entry_point()
    well_count = 6
    files_pr_well = 3
    assert len(list(os.walk(outputdirectory))[0][2]) == well_count * files_pr_well
    assert os.path.exists("csvfile.csv")  # Should be independent of --outputdirectory


def test_gendata_rft_entry_point_wrong_well_file(
    tmpdir, norne_data, monkeypatch, capsys
):

    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]
    with open("well_and_time.txt", "w+") as fh:
        fh.write("NO_FILE_HERE 1 12 2005 0\n")
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit, match="NO_FILE_HERE.txt not found"):
        main_entry_point()


def test_gendata_rft_entry_point(tmpdir, norne_data, monkeypatch):

    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main_entry_point()
    expected_results_dir = os.path.join(tmpdir.strpath, "expected_results")

    expected_files = [
        os.path.join(EXPECTED_RESULTS_PATH_NORNE, fname)
        for fname in os.listdir(EXPECTED_RESULTS_PATH_NORNE)
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

            conftest._assert_almost_equal_line_by_line(expected_file, result_file)


def test_gendata_inactive_info_point_not_in_grid(tmpdir, norne_data, monkeypatch):

    with open("B-1AH.txt", "a+") as fh:
        fh.write("0 1 2 3\n")

    with open("well_and_time.txt", "w+") as fh:
        fh.write("B-1AH 1 12 2005 0\n")

    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main_entry_point()

    with open("RFT_B-1AH_0_inactive_info") as fh:
        result = fh.read()
        assert result.startswith(
            "TRAJECTORY_POINT_NOT_IN_GRID (utm_x=0.0, utm_y=1.0, measured_depth=2.0)"
        )


def test_gendata_inactive_info_zone_mismatch(tmpdir, norne_data, monkeypatch):

    with open("well_and_time.txt", "w+") as fh:
        fh.write("B-1AH 1 12 2005 0\n")

    with open(os.path.join(tmpdir.strpath, "B-1AH.txt"), "r+") as fh:
        lines = fh.readlines()

    line = lines[-1].rsplit(" ", 1)[0]
    line = line + " last_zone"

    with open("B-1AH.txt", "w+") as fh:
        fh.write(line)

    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main_entry_point()

    with open("RFT_B-1AH_0_inactive_info") as fh:
        result = fh.read()
        assert result.startswith("ZONE_MISMATCH (utm_x=")


def test_gendata_inactive_info_not_in_rft(tmpdir, norne_data, monkeypatch):

    with open("well_and_time.txt", "w+") as fh:
        fh.write("B-1AH 1 12 2005 0\n")

    with open(os.path.join(tmpdir.strpath, "B-1AH.txt"), "r+") as fh:
        lines = fh.readlines()

    line = lines[-1].rsplit(" ", 3)[0]
    line += " 2700 2700"

    with open("B-1AH.txt", "w+") as fh:
        fh.write(line)

    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main_entry_point()

    with open("RFT_B-1AH_0_inactive_info") as fh:
        result = fh.read()
        assert result.startswith("TRAJECTORY_POINT_NOT_IN_RFT (utm_x=")


def test_gendata_inactive_info_zone_missing_value(tmpdir, norne_data, monkeypatch):

    with open("well_and_time.txt", "w+") as fh:
        fh.write("B-1AH 1 12 2005 0\n")

    with open("zonemap.txt", "w+") as fh:
        fh.write("1 zone1\n")

    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main_entry_point()

    with open("RFT_B-1AH_0_inactive_info") as fh:
        result = fh.read()
        assert result.startswith("ZONEMAP_MISSING_VALUE (utm_x=")


def test_partial_rft_file(tmpdir, norne_data, monkeypatch, caplog):
    """Test how the code behaves when some report steps are missing, e.g.
    a Eclipse simulation that has crashed midway.

    We emulate this situation by asking for well-times that are not in the
    binary RFT output in the test dataset.

    In such a situation, we do not want the OK file to written,
    but the CSV file can contain results of data up until the Eclipse crash.
    """

    # Append an extra non-existing date to the well_and_time.txt test-file
    with open("well_and_time.txt", "a") as file_h:
        file_h.write("B-1AH 1 12 2045 0")

    csv_filename = "gendata_rft.csv"
    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
        "-w",
        "well_and_time.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
        "-c",
        csv_filename,
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit):
        main_entry_point()

    assert "Failed to extract all requested RFT data" in caplog.text

    assert os.path.exists(csv_filename)
    assert not pd.read_csv(csv_filename).empty
    assert not os.path.exists("GENDATA_RFT.OK")


def test_one_wrong_date(tmpdir, norne_data, monkeypatch, caplog):
    with open("well_wrongtime.txt", "w") as file_h:
        file_h.write("B-1AH 1 12 2045 0")

    csv_filename = "gendata_rft.csv"
    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
        "-w",
        "well_wrongtime.txt",
        "-t",
        tmpdir.strpath,
        "-z",
        "zonemap.txt",
        "-c",
        csv_filename,
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit):
        main_entry_point()

    assert "Failed to extract all requested RFT data" in caplog.text

    assert not os.path.exists(csv_filename)
    assert not os.path.exists("GENDATA_RFT.OK")


def test_empty_well_and_time(tmpdir, norne_data, monkeypatch, caplog):
    def file_count_cwd():
        return len(list(os.walk("."))[0][2])

    with open("empty.txt", "w") as file_h:
        file_h.write("")

    arguments = [
        "script_name",
        "-e",
        ECL_BASE_NORNE,
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
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit):
        main_entry_point()
    assert "No RFT data requested" in caplog.text

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
    for line in Path(job_description_file).read_text().split("\n"):
        if line.startswith("DEFAULT"):
            if line.split()[0:2] == ["DEFAULT", "<CSVFILE>"]:
                csv_job_default = line.split()[2]
            if line.split()[0:2] == ["DEFAULT", "<OUTPUTDIRECTORY>"]:
                directory_job_default = line.split()[2]

    # And compare with argparse:
    assert csv_job_default == _build_parser().get_default("csvfile")
    assert directory_job_default == _build_parser().get_default("outputdirectory")
