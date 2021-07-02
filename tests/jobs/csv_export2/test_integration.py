# pylint: disable=unsubscriptable-object  # pylint issue
import os
import sys
import subprocess

import pytest
import rstcheck
import pandas as pd

from semeio.workflows.csv_export2 import csv_export2

from tests.jobs.csv_export2 import conftest

test_header = [
    "ENSEMBLE",
    "REAL",
    "DATE",
    "FOPR",
    "X",
    "FLUID_PARAMS:SWL",
    "FLUID_PARAMS:SGCR",
    "A",
    "H",
]

NORNE_VECS = ["FGPT", "FLPT", "FOPT", "FVPT", "FWPT"]


@pytest.mark.skipif(
    conftest.find_available_test_data() is None, reason="no ert-statoil test-data"
)
@pytest.mark.usefixtures("ert_statoil_test_data")
def test_failed_realization_no_summary_file(ert_statoil_test_data):
    path_file = "pathfile_path_to_failed_realizations.txt"
    export_file = "export.txt"
    csv_export2.csv_exporter(
        runpathfile=path_file,
        time_index="monthly",
        outputfile=export_file,
        column_keys="FOPR",
    )
    verifyExportedFile(
        export_file,
        test_header,
        {("iter-1", 1), ("iter-2", 2), ("iter-1", 0), ("iter-1", 2)},
    )


@pytest.mark.skipif(
    conftest.find_available_test_data() is None, reason="no ert-statoil test-data"
)
@pytest.mark.usefixtures("ert_statoil_test_data")
def test_one_iteration(ert_statoil_test_data):
    path_file = "pathfile.txt"
    export_file = "export.txt"
    csv_export2.csv_exporter(
        runpathfile=path_file,
        time_index="monthly",
        outputfile=export_file,
        column_keys="FOPR",
    )
    verifyExportedFile(
        export_file, test_header, {("iter-1", 0), ("iter-1", 1), ("iter-1", 2)}
    )


@pytest.mark.skipif(
    conftest.find_available_test_data() is None, reason="no ert-statoil test-data"
)
@pytest.mark.usefixtures("ert_statoil_test_data")
def test_missing_realization(ert_statoil_test_data):
    path_file = "pathfile2.txt"
    export_file = "export.txt"
    csv_export2.csv_exporter(
        runpathfile=path_file,
        time_index="monthly",
        outputfile=export_file,
        column_keys="FOPR",
    )
    verifyExportedFile(export_file, test_header, {("iter-1", 2), ("iter-1", 0)})


@pytest.mark.skipif(
    conftest.find_available_test_data() is None, reason="no ert-statoil test-data"
)
@pytest.mark.usefixtures("ert_statoil_test_data")
def test_iterations(ert_statoil_test_data):
    path_file = "pathfile3.txt"
    export_file = "export.txt"
    csv_export2.csv_exporter(
        runpathfile=path_file,
        time_index="monthly",
        outputfile=export_file,
        column_keys="FOPR",
    )
    verifyExportedFile(
        export_file,
        test_header,
        {
            ("iter-1", 0),
            ("iter-1", 1),
            ("iter-1", 2),
            ("iter-2", 0),
            ("iter-2", 1),
            ("iter-2", 2),
        },
    )


@pytest.mark.skipif(
    conftest.find_available_test_data() is None, reason="no ert-statoil test-data"
)
@pytest.mark.usefixtures("ert_statoil_test_data")
def test_no_iterations(ert_statoil_test_data):
    path_file = "pathfile1.txt"
    export_file = "export.txt"

    with pytest.raises(KeyError):
        csv_export2.csv_exporter(
            runpathfile=path_file,
            time_index="monthly",
            outputfile=export_file,
            column_keys="FOPR",
        )


@pytest.mark.parametrize("input_rst", [csv_export2.DESCRIPTION, csv_export2.EXAMPLES])
def test_valid_rst(input_rst):
    """
    Check that the documentation passed through the plugin system is
    valid rst
    """
    assert not list(rstcheck.check(input_rst))


@pytest.mark.usefixtures("norne_mocked_ensembleset")
def test_norne_ensemble(norne_mocked_ensembleset):
    csv_export2.csv_exporter(
        runpathfile="runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["F?PT"],
    )
    verifyExportedFile(
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
def test_norne_ensemble_noparams(norne_mocked_ensembleset_noparams):
    csv_export2.csv_exporter(
        runpathfile="runpathfile",
        time_index="yearly",
        outputfile="unsmry--yearly.csv",
        column_keys=["FOPT"],
    )
    verifyExportedFile(
        "unsmry--yearly.csv",
        ["ENSEMBLE", "REAL", "DATE", "FOPT"],
        {
            ("iter-0", 0),
            ("iter-0", 1),
            ("iter-1", 0),
            ("iter-1", 1),
        },
    )


def verifyExportedFile(exported_file_name, result_header, result_iter_rel):
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


@pytest.mark.integration
def test_ert_integration(norne_mocked_ensembleset):
    """Mock an ERT config and test the workflow"""
    with open("FOO.DATA", "w") as file_h:
        file_h.write("--Empty")

    with open("wf_csvexport", "w") as file_h:
        file_h.write(
            (
                # This workflow is representing the example in csv_export2.py:
                (
                    "MAKE_DIRECTORY csv_output\n"
                    "EXPORT_RUNPATH * | *\n"  # (not really relevant in mocked case)
                    "CSV_EXPORT2 runpathfile csv_output/data.csv monthly FOPT\n"
                    # Example in documentation uses <RUNPATH_FILE> which is
                    # linked to the RUNPATH keyword that we don't use in this
                    # test (mocking data gets more complex if that is to be used)
                )
            )
        )

    ert_config = [
        "ECLBASE FOO.DATA",
        "QUEUE_SYSTEM LOCAL",
        "NUM_REALIZATIONS 2",
        "LOAD_WORKFLOW wf_csvexport",
        "HOOK_WORKFLOW wf_csvexport PRE_SIMULATION",
    ]

    ert_config_fname = "test.ert"
    with open(ert_config_fname, "w") as file_h:
        file_h.write("\n".join(ert_config))

    subprocess.run(["ert", "test_run", ert_config_fname], check=True)

    assert pd.read_csv("csv_output/data.csv").shape == (16, 5)


@pytest.mark.skipif(sys.version_info < (3, 7), reason="Py 3.7 for capture_output")
@pytest.mark.integration
def test_ert_integration_errors(norne_mocked_ensembleset):
    """Test CSV_EXPORT2 when runpathfile points to non-existing realizations

    This test proves that CSV_EXPORT2 happily skips non-existing
    realizations, but emits a warning that there is no STATUS file.
    """
    with open("FOO.DATA", "w") as file_h:
        file_h.write("--Empty")

    # Append a not-existing realizations to the runpathfile:
    with open("runpathfile", "a") as file_h:
        file_h.write("002 realization-2/iter-0 NORNE_1 000")

    with open("wf_csvexport", "w") as file_h:
        file_h.write("CSV_EXPORT2 runpathfile data.csv monthly FOPT\n")

    ert_config = [
        "ECLBASE FOO.DATA",
        "QUEUE_SYSTEM LOCAL",
        "NUM_REALIZATIONS 2",
        "LOAD_WORKFLOW wf_csvexport",
        "HOOK_WORKFLOW wf_csvexport PRE_SIMULATION",
    ]

    ert_config_fname = "test.ert"
    with open(ert_config_fname, "w") as file_h:
        file_h.write("\n".join(ert_config))

    ertoutput = subprocess.run(
        ["ert", "test_run", ert_config_fname], check=True, capture_output=True
    )

    assert "fmu.ensemble.realization - WARNING - No STATUS file" in str(
        ertoutput.stdout
    )
    assert "realization-2/iter-0" in str(ertoutput.stdout)

    assert os.path.exists("data.csv")
    assert pd.read_csv("data.csv").shape == (16, 5)
