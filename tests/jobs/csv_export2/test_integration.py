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
