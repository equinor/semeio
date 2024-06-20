import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy
import pandas as pd
import pytest

from semeio.forward_models.scripts.gendata_rft import main_entry_point
from tests.forward_models.rft import conftest

ECL_BASE_NORNE = conftest.get_ecl_base_norne()
ECL_BASE_REEK = conftest.get_ecl_base_reek()
MOCK_DATA_CONTENT_NORNE = conftest.get_mock_data_content_norne()
EXPECTED_RESULTS_PATH_NORNE = conftest.get_expected_results_path_norne()


@pytest.mark.usefixtures("norne_data")
def test_gendata_rft_csv(tmpdir, monkeypatch):
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
        "report_step",
    }
    assert required_columns.issubset(set(dframe.columns))
    assert len(dframe["well"].unique()) == len(MOCK_DATA_CONTENT_NORNE)

    assert set(dframe["report_step"]) == {0}

    # i-j-k always non-null together:
    assert (dframe["i"].isnull() == dframe["j"].isnull()).all()
    assert (dframe["i"].isnull() == dframe["k"].isnull()).all()

    # inactive_info must be non-null when active = False
    assert (~dframe[~dframe["is_active"]]["inactive_info"].isnull()).all()


@pytest.mark.usefixtures("reek_data")
def test_gendata_rft_csv_reek(tmpdir, monkeypatch):
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
    assert numpy.isclose(dframe["pressure"].to_numpy()[0], 304.37)
    assert numpy.isclose(dframe["swat"].to_numpy()[0], 0.151044)
    assert numpy.isclose(dframe["soil"].to_numpy()[0], 1 - 0.151044)
    assert numpy.isclose(dframe["sgas"].to_numpy()[0], 0.0)

    # This requires 1-indexed ijk in the dataframe:
    assert dframe["i"].to_numpy()[0] == 29
    assert dframe["j"].to_numpy()[0] == 28
    assert dframe["k"].to_numpy()[0] == 8


@pytest.mark.usefixtures("norne_data")
def test_gendata_rft_directory(tmpdir, monkeypatch):
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


@pytest.mark.usefixtures("norne_data")
def test_gendata_rft_entry_point_wrong_well_file(tmpdir, monkeypatch):
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
    with open("well_and_time.txt", "w+", encoding="utf-8") as file:
        file.write("NO_FILE_HERE 2005-12-01 0\n")
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit, match="NO_FILE_HERE.txt not found"):
        main_entry_point()


@pytest.mark.usefixtures("norne_data")
def test_gendata_rft_entry_point(tmpdir, monkeypatch):
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

            _assert_almost_equal_line_by_line(expected_file, result_file)


@pytest.mark.usefixtures("reek_data")
def test_multiple_report_steps(tmpdir, monkeypatch):
    with open("well_and_time.txt", "w+", encoding="utf-8") as file:
        file.write("OP_1 2000-02-01 0\n")
        file.write("OP_1 2001-01-01 1\n")

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
    csvdata = pd.read_csv("gendata_rft.csv")
    assert (csvdata["report_step"].to_numpy() == [0, 1]).all()
    assert (csvdata["time"].to_numpy() == ["2000-02-01", "2001-01-01"]).all()
    assert numpy.isclose(csvdata["pressure"].to_numpy(), [304.370087, 249.214965]).all()

    # Testing only the data for report step 1:
    assert numpy.isclose(
        float(Path("RFT_OP_1_1").read_text(encoding="utf8")), 249.214965
    )
    assert Path("RFT_OP_1_1_active").read_text(encoding="utf8") == "1"
    assert not Path("RFT_OP_1_1_inactive_info").read_text(encoding="utf8")
    assert numpy.isclose(float(Path("RFT_SGAS_OP_1_1").read_text(encoding="utf8")), 0.0)
    assert numpy.isclose(
        float(Path("RFT_SOIL_OP_1_1").read_text(encoding="utf8")), 0.6119158
    )
    assert numpy.isclose(
        float(Path("RFT_SWAT_OP_1_1").read_text(encoding="utf8")), 0.3880841
    )


@pytest.mark.usefixtures("norne_data")
def test_gendata_inactive_info_point_not_in_grid(tmpdir, monkeypatch):
    with open("B-1AH.txt", "a+", encoding="utf-8") as file:
        file.write("0 1 2 3\n")

    with open("well_and_time.txt", "w+", encoding="utf-8") as file:
        file.write("B-1AH 2005-12-01 0\n")

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

    with open("RFT_B-1AH_0_inactive_info", encoding="utf-8") as file:
        result = file.read()
        assert result.startswith(
            "TRAJECTORY_POINT_NOT_IN_GRID (utm_x=0.0, utm_y=1.0, measured_depth=2.0)"
        )


@pytest.mark.usefixtures("norne_data")
def test_gendata_inactive_info_zone_mismatch(tmpdir, monkeypatch):
    with open("well_and_time.txt", "w+", encoding="utf-8") as file:
        file.write("B-1AH 2005-12-01 0\n")

    with open(
        os.path.join(tmpdir.strpath, "B-1AH.txt"), "r+", encoding="utf-8"
    ) as file:
        lines = file.readlines()

    line = lines[-1].rsplit(" ", 1)[0]
    line += " last_zone"

    with open("B-1AH.txt", "w+", encoding="utf-8") as file:
        file.write(line)

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

    with open("RFT_B-1AH_0_inactive_info", encoding="utf-8") as file:
        result = file.read()
        assert result.startswith("ZONE_MISMATCH (utm_x=")


@pytest.mark.usefixtures("norne_data")
def test_gendata_inactive_info_not_in_rft(tmpdir, monkeypatch):
    with open("well_and_time.txt", "w+", encoding="utf-8") as file:
        file.write("B-1AH 2005-12-01 0\n")

    with open(
        os.path.join(tmpdir.strpath, "B-1AH.txt"), "r+", encoding="utf-8"
    ) as file:
        lines = file.readlines()

    line = lines[-1].rsplit(" ", 3)[0]
    line += " 2700 2700"

    with open("B-1AH.txt", "w+", encoding="utf-8") as file:
        file.write(line)

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

    with open("RFT_B-1AH_0_inactive_info", encoding="utf-8") as file:
        result = file.read()
        assert result.startswith("TRAJECTORY_POINT_NOT_IN_RFT (utm_x=")


@pytest.mark.usefixtures("norne_data")
def test_gendata_inactive_info_zone_missing_value(tmpdir, monkeypatch):
    with open("well_and_time.txt", "w+", encoding="utf-8") as file:
        file.write("B-1AH 2005-12-01 0\n")

    with open("zonemap.txt", "w+", encoding="utf-8") as file:
        file.write("1 zone1\n")

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

    with open("RFT_B-1AH_0_inactive_info", encoding="utf-8") as file:
        result = file.read()
        assert result.startswith("ZONEMAP_MISSING_VALUE (utm_x=")


@pytest.mark.usefixtures("norne_data")
def test_partial_rft_file(tmpdir, monkeypatch, caplog):
    """Test how the code behaves when some report steps are missing, e.g.
    a Eclipse simulation that has crashed midway.

    We emulate this situation by asking for well-times that are not in the
    binary RFT output in the test dataset.

    In such a situation, we do not want the OK file to written,
    but the CSV file can contain results of data up until the Eclipse crash.
    """

    # Append an extra non-existing date to the well_and_time.txt test-file
    with open("well_and_time.txt", "a", encoding="utf-8") as file_h:
        file_h.write("B-1AH 2045-12-01 0")

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


@pytest.mark.usefixtures("norne_data")
def test_one_wrong_date(tmpdir, monkeypatch, caplog):
    with open("well_wrongtime.txt", "w", encoding="utf-8") as file_h:
        file_h.write("B-1AH 2045-12-01 0")

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


@pytest.mark.usefixtures("norne_data")
def test_empty_well_and_time(tmpdir, monkeypatch, caplog):
    def file_count_cwd():
        return len(list(os.walk("."))[0][2])

    with open("empty.txt", "w", encoding="utf-8") as file_h:
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


@pytest.mark.ert_integration
def test_ert_setup_one_well_one_rft_point(tmpdir):
    """Test a simple ERT integration of GENDATA_RFT with one RFT point at one
    time instant.

    This test should confirm that the GENDATA_RFT + GEN_DATA works together, and
    acts as a reference for documentation. This tests intentionally fails the
    ERT update step.
    """
    tmpdir.chdir()
    Path("realization-0/iter-0").mkdir(parents=True)
    Path("realization-0/iter-1").mkdir(parents=True)
    for filename in Path(ECL_BASE_REEK).parent.glob("2_R001_REEK-0.*"):
        shutil.copy(filename, Path("realization-0/iter-0") / filename.name)
        shutil.copy(filename, Path("realization-0/iter-1") / filename.name)

    Path("rft_input").mkdir()

    # Write the well trajectory:
    (Path("rft_input") / "OP_1.txt").write_text(
        "462608.57 5934210.96 1644.38 1624.38 zone2", encoding="utf-8"
    )

    # List the well and dates for which we have observations:
    (Path("rft_input") / "well_time.txt").write_text(
        "OP_1 2000-02-01 1", encoding="utf-8"
    )

    Path("observations").mkdir()

    # Write observation file:
    Path("observations.txt").write_text(
        "GENERAL_OBSERVATION OP_1_OBS1 "
        "{DATA=OP_1_RFT_SIM1; RESTART=1; OBS_FILE=observations/OP_1_1.obs; };\n",
        encoding="utf-8",
    )

    # Chosen filename syntax for obs file: <wellname>_<report_step>.obs
    Path("observations/OP_1_1.obs").write_text("304 1", encoding="utf-8")
    # measured pressure, abs error

    # Time-map needed for loading observations with no summary file. It seems we need
    # one date for start-date, and one date for every report_step in use.
    Path("time_map.txt").write_text("2000-01-01\n2000-02-01", encoding="utf-8")

    Path("parameter_prior").write_text("PARAM_1 UNIFORM 0 1\n", encoding="utf-8")
    Path("parameter_template").write_text('{\n"a": <PARAM_1>\n}', encoding="utf-8")
    # Write an ERT config file
    # pylint: disable=line-too-long
    Path("config.ert").write_text(
        """
RUNPATH realization-<IENS>/iter-<ITER>
ECLBASE 2_R001_REEK-%d
TIME_MAP time_map.txt
QUEUE_SYSTEM LOCAL
NUM_REALIZATIONS 1
OBS_CONFIG observations.txt

FORWARD_MODEL GENDATA_RFT(<PATH_TO_TRAJECTORY_FILES>=../../rft_input, <WELL_AND_TIME_FILE>=../../rft_input/well_time.txt)
GEN_KW PARAMS parameter_template parameters.json parameter_prior
GEN_DATA OP_1_RFT_SIM1 INPUT_FORMAT:ASCII REPORT_STEPS:1 RESULT_FILE:RFT_OP_1_%d
""",  # noqa
        encoding="utf-8",
    )

    # pylint: disable=subprocess-run-check
    # (assert on the return code further down)
    result = subprocess.run(
        ["ert", "ensemble_smoother", "--target-case", "default_%d", "config.ert"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdouterr = result.stdout.decode() + result.stderr.decode()

    if result.returncode != 0:
        print(stdouterr)
        # Print any stderr files from the forward model:
        for stderrfile in Path("realization-0").glob("iter-*/*stderr*"):
            if stderrfile.stat().st_size > 0:
                print(stderrfile)
                print(stderrfile.read_text())

    # Verify that the configuration of time_map vs observations.txt match:
    assert "failed to load observation data from" not in stdouterr

    # These can be triggered by various GEN_DATA errors:
    assert "Failed load data for GEN_DATA node" not in stdouterr
    assert "gen_obs_assert_data_size: size mismatch" not in stdouterr
    assert "conf_instance_get_restart_nr" not in stdouterr

    # The purpose of the test is to demonstrate and run through the GEN_DATA
    # forward model without errors, but it will fail in the update step as this
    # test is not constructed for that:
    assert "Experiment failed" in stdouterr

    # Asserts on GENDATA_RFT output:
    assert Path("realization-0/iter-0/RFT_OP_1_1").is_file()

    assert Path("realization-0/iter-0/OK").is_file()


@pytest.mark.ert_integration
def test_ert_setup_one_well_two_points_different_time_and_depth(tmpdir):
    """Test a simple ERT integration of GENDATA_RFT with one RFT point at one
    time instant and another RFT point at another time and another depth.

    This test should confirm that the GENDATA_RFT + GEN_DATA works together, and
    acts as a reference for documentation. This tests intentionally fails the
    ERT update step.

    Also using a dedicated directory for RUNPATH output of rft data.
    """
    tmpdir.chdir()
    Path("realization-0/iter-0").mkdir(parents=True)
    Path("realization-0/iter-1").mkdir(parents=True)
    for filename in Path(ECL_BASE_REEK).parent.glob("2_R001_REEK-0.*"):
        shutil.copy(filename, Path("realization-0/iter-0") / filename.name)
        shutil.copy(filename, Path("realization-0/iter-1") / filename.name)

    Path("rft_input").mkdir()

    # Write the well trajectory:
    (Path("rft_input") / "OP_1.txt").write_text(
        "462608.57 5934210.96 1644.38 1624.38 zone2\n"
        "462608.57 5934210.96 1644.38 1634.38 zone2\n"
    )

    # List the well and dates for which we have observations:
    (Path("rft_input") / "well_time.txt").write_text(
        "OP_1 2000-02-01 1\nOP_1 2001-01-01 2"
    )

    Path("observations").mkdir()

    # Write observation file:
    Path("observations.txt").write_text(
        "GENERAL_OBSERVATION OP_1_OBS1 "
        "{DATA=OP_1_RFT_SIM1; RESTART=1; OBS_FILE=observations/OP_1_1.obs; };\n"
        "GENERAL_OBSERVATION OP_1_OBS2 "
        "{DATA=OP_1_RFT_SIM2; RESTART=2; OBS_FILE=observations/OP_1_2.obs; };\n",
        encoding="utf-8",
    )

    # Chosen filename syntax for obs file: <wellname>_<report_step>.obs
    Path("observations/OP_1_1.obs").write_text("304 1\n-1 0.1\n", encoding="utf-8")
    Path("observations/OP_1_2.obs").write_text("-1 0.1\n249 1\n", encoding="utf-8")
    # measured pressure, abs error

    # Time-map needed for loading observations with no summary file. It seems we need
    # one date for start-date, and one date for every report_step in use.
    Path("time_map.txt").write_text(
        "2000-01-01\n2000-02-01\n2001-01-01", encoding="utf-8"
    )

    Path("parameter_prior").write_text("PARAM_1 UNIFORM 0 1\n", encoding="utf-8")
    Path("parameter_template").write_text('{\n"a": <PARAM_1>\n}', encoding="utf-8")

    # Write an ERT config file
    # pylint: disable=line-too-long
    Path("config.ert").write_text(
        dedent(
            """
    RUNPATH realization-<IENS>/iter-<ITER>
    ECLBASE 2_R001_REEK-%d
    TIME_MAP time_map.txt
    QUEUE_SYSTEM LOCAL
    NUM_REALIZATIONS 1
    OBS_CONFIG observations.txt

    FORWARD_MODEL MAKE_DIRECTORY(<DIRECTORY>=gendata_rft)
    FORWARD_MODEL GENDATA_RFT(<PATH_TO_TRAJECTORY_FILES>=../../rft_input, <WELL_AND_TIME_FILE>=../../rft_input/well_time.txt, <OUTPUTDIRECTORY>=gendata_rft)

    GEN_DATA OP_1_RFT_SIM1 INPUT_FORMAT:ASCII REPORT_STEPS:1 RESULT_FILE:gendata_rft/RFT_OP_1_%d
    GEN_DATA OP_1_RFT_SIM2 INPUT_FORMAT:ASCII REPORT_STEPS:2 RESULT_FILE:gendata_rft/RFT_OP_1_%d
    GEN_KW PARAMS parameter_template parameters.json parameter_prior
    """  # noqa
        ),
        encoding="utf-8",
    )

    # pylint: disable=subprocess-run-check
    # (assert on the return code further down)
    result = subprocess.run(
        ["ert", "ensemble_smoother", "--target-case", "default_%d", "config.ert"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdouterr = result.stdout.decode() + result.stderr.decode()

    if result.returncode != 0:
        print(stdouterr)
        # Print any stderr files from the forward model:
        for stderrfile in Path("realization-0").glob("iter-*/*stderr*"):
            if stderrfile.stat().st_size > 0:
                print(stderrfile)
                print(stderrfile.read_text())

    # Fails if time_map vs observations.txt does not match:
    assert "failed to load observation data from" not in stdouterr

    # These can be triggered by GEN_DATA errors:
    assert "Failed load data for GEN_DATA node" not in stdouterr
    assert "gen_obs_assert_data_size: size mismatch" not in stdouterr

    # The purpose of the test is to demonstrate and run through the GEN_DATA
    # forward model without errors, but it will fail in the update step as this
    # test is not constructed for that:
    assert "No active observations" in stdouterr
    assert "Experiment failed" in stdouterr

    # Asserts on GENDATA_RFT output:
    assert Path("realization-0/iter-0/gendata_rft/RFT_OP_1_1").is_file()

    assert Path("realization-0/iter-0/OK").is_file()


def _assert_almost_equal_line_by_line(file1, file2):
    with open(file1, encoding="utf-8") as file:
        file1_content = file.readlines()

    with open(file2, encoding="utf-8") as file:
        file2_content = file.readlines()

    assert len(file1_content) == len(file2_content)

    for line1, line2 in zip(file1_content, file2_content):
        try:
            line1, line2 = float(line1), float(line2)
        except ValueError:
            continue
        numpy.testing.assert_almost_equal(line1, line2, decimal=7)
