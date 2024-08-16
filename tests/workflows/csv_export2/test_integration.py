import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import rstcheck_core.checker

from semeio.workflows.csv_export2 import csv_export2

NORNE_VECS = ["FGPT", "FLPT", "FOPT", "FVPT", "FWPT"]


@pytest.mark.usefixtures("norne_mocked_ensembleset")
def test_that_a_not_found_realization_is_skipped():
    shutil.rmtree("realization-1/iter-1")
    csv_export2.csv_exporter(
        runpathfile="runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["F?PT"],
    )
    verify_exported_file(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE"] + NORNE_VECS + ["FOO"],
        {
            ("iter-0", 0),
            ("iter-0", 1),
            ("iter-1", 0),
        },
    )


@pytest.mark.usefixtures("norne_mocked_ensembleset")
def test_that_a_failed_realization_is_skipped():
    os.remove("realization-0/iter-1/NORNE_0.SMSPEC")
    csv_export2.csv_exporter(
        runpathfile="runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["F?PT"],
    )
    verify_exported_file(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE"] + NORNE_VECS + ["FOO"],
        {
            ("iter-0", 0),
            ("iter-0", 1),
            ("iter-1", 1),
        },
    )


@pytest.mark.usefixtures("norne_mocked_ensembleset")
def test_that_a_missing_realization_index_is_ok():
    rp_lines = Path("runpathfile").read_text(encoding="utf-8").splitlines()
    Path("sliced_runpathfile").write_text(
        rp_lines[1] + "\n" + rp_lines[3], encoding="utf-8"
    )
    csv_export2.csv_exporter(
        runpathfile="sliced_runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["F?PT"],
    )
    verify_exported_file(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE"] + NORNE_VECS + ["FOO"],
        {
            ("iter-0", 1),
            ("iter-1", 1),
        },
    )


@pytest.mark.usefixtures("norne_mocked_ensembleset")
def test_that_iterations_in_runpathfile_cannot_be_defaulted():
    shutil.move("realization-0/iter-0", "real0")
    shutil.move("realization-1/iter-0", "real1")
    shutil.rmtree("realization-0")
    shutil.rmtree("realization-1")
    Path("runpathfile").write_text(
        "000 real0 NORNE_0\n001 real1 NORNE_1\n", encoding="utf-8"
    )

    with pytest.raises(UserWarning):
        csv_export2.csv_exporter(
            runpathfile="runpathfile",
            time_index="yearly",
            outputfile="unsmry--yearly.csv",
            column_keys=["F?PT"],
        )


def test_empty_file_yields_user_warning():
    with open("empty_file", "a", encoding="utf-8") as empty_file, pytest.raises(
        UserWarning, match="No data found"
    ):
        csv_export2.csv_exporter(
            runpathfile=empty_file.name,
            time_index="raw",
            outputfile="unsmry--yearly.csv",
            column_keys=["*"],
        )


@pytest.mark.parametrize("input_rst", [csv_export2.DESCRIPTION, csv_export2.EXAMPLES])
def test_valid_rst(input_rst):
    """
    Check that the documentation passed through the plugin system is
    valid rst
    """
    assert not list(rstcheck_core.checker.check_source(input_rst))


@pytest.mark.usefixtures("norne_mocked_ensembleset")
def test_norne_ensemble():
    csv_export2.csv_exporter(
        runpathfile="runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["F?PT"],
    )
    verify_exported_file(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE"] + NORNE_VECS + ["FOO"],
        {
            ("iter-0", 0),
            ("iter-0", 1),
            ("iter-1", 0),
            ("iter-1", 1),
        },
    )


@pytest.mark.usefixtures("norne_mocked_ensembleset_noparams")
def test_norne_ensemble_noparams():
    csv_export2.csv_exporter(
        runpathfile="runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["FOPT"],
    )
    verify_exported_file(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE", "FOPT"],
        {
            ("iter-0", 0),
            ("iter-0", 1),
            ("iter-1", 0),
            ("iter-1", 1),
        },
    )


def verify_exported_file(exported_file_name, result_header, result_iter_rel):
    """Verify an exported CSV file with respect to:

        * Exactly the set of requested headers is found
        * The realizations and iterations that exist must equal
          given set of tuples.

    Args:
        exported_file_name (str): path to CSV file.
        result_header (list of str): The strings required in the header.
        result_iter_real (set): Set of 2-tuples: {(iterstring, realidx)}
    """
    dframe = pd.read_csv(exported_file_name)
    assert set(dframe.columns) == set(result_header)
    assert (
        set(dframe[["ENSEMBLE", "REAL"]].itertuples(index=False, name=None))
        == result_iter_rel
    )


@pytest.mark.ert_integration
@pytest.mark.usefixtures("norne_mocked_ensembleset")
def test_ert_integration():
    """Mock an ERT config and test the workflow"""
    with open("FOO.DATA", "w", encoding="utf-8") as file_h:
        file_h.write("--Empty")

    with open("wf_csvexport", "w", encoding="utf-8") as file_h:
        file_h.write(
            # This workflow is representing the example in csv_export2.py:
            "MAKE_DIRECTORY csv_output\n"
            "EXPORT_RUNPATH * | *\n"  # (not really relevant in mocked case)
            "CSV_EXPORT2 runpathfile csv_output/data.csv monthly FOPT\n"
            # Example in documentation uses <RUNPATH_FILE> which is
            # linked to the RUNPATH keyword that we don't use in this
            # test (mocking data gets more complex if that is to be used)
        )

    ert_config = [
        "ECLBASE FOO.DATA",
        "QUEUE_SYSTEM LOCAL",
        "NUM_REALIZATIONS 2",
        "LOAD_WORKFLOW wf_csvexport",
        "HOOK_WORKFLOW wf_csvexport PRE_SIMULATION",
    ]

    ert_config_fname = "test.ert"
    with open(ert_config_fname, "w", encoding="utf-8") as file_h:
        file_h.write("\n".join(ert_config))

    subprocess.run(["ert", "test_run", ert_config_fname], check=True)

    assert pd.read_csv("csv_output/data.csv").shape == (16, 5)


@pytest.mark.ert_integration
@pytest.mark.usefixtures("norne_mocked_ensembleset")
def test_ert_integration_errors(snapshot):
    """Test CSV_EXPORT2 when runpathfile points to non-existing realizations

    This test proves that CSV_EXPORT2 happily skips non-existing
    realizations, but emits a warning that there is no STATUS file.
    """
    with open("FOO.DATA", "w", encoding="utf-8") as file_h:
        file_h.write("--Empty")

    # Append a not-existing realizations to the runpathfile:
    with open("runpathfile", "a", encoding="utf-8") as file_h:
        file_h.write("002 realization-2/iter-0 NORNE_1 000")

    with open("wf_csvexport", "w", encoding="utf-8") as file_h:
        file_h.write("CSV_EXPORT2 runpathfile data.csv monthly FOPT\n")

    ert_config = [
        "ECLBASE FOO.DATA",
        "QUEUE_SYSTEM LOCAL",
        "NUM_REALIZATIONS 2",
        "LOAD_WORKFLOW wf_csvexport",
        "HOOK_WORKFLOW wf_csvexport PRE_SIMULATION",
    ]

    ert_config_fname = "test.ert"
    with open(ert_config_fname, "w", encoding="utf-8") as file_h:
        file_h.write("\n".join(ert_config))

    subprocess.run(["ert", "test_run", ert_config_fname], check=True)

    log_file = next(Path("logs").glob("ert-log*txt"))
    ertlog = log_file.read_text(encoding="utf-8")

    assert "No STATUS file" in ertlog
    assert "realization-2/iter-0" in ertlog

    assert os.path.exists("data.csv")
    data = pd.read_csv("data.csv")
    snapshot.assert_match(
        data.to_csv(lineterminator="\n"),
        "csv_data.csv",
    )
