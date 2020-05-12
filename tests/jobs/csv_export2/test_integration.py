import pytest
from semeio.jobs.csv_export2 import csv_export2

from tests.jobs.csv_export2 import conftest

test_header = "ENSEMBLE,REAL,DATE,FOPR,X,FLUID_PARAMS:SWL,FLUID_PARAMS:SGCR,A,H\n"


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


def verifyExportedFile(exported_file_name, result_header, result_iter_rel):
    with open(exported_file_name, "r") as exported_file:
        lines = exported_file.readlines()

        assert sorted(lines[0]) == sorted(result_header)

        iter_rel_no = set()

        for index, line in enumerate(lines[1:]):
            tokens = line.split(",")
            iteration, rel = tokens[0], int(tokens[1])

            if (iteration, rel) not in iter_rel_no:
                iter_rel_no.add((iteration, rel))

        assert iter_rel_no == result_iter_rel
