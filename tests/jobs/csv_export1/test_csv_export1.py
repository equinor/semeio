import sys

import pytest
from semeio.jobs.csv_export1 import csv_export1

from tests.jobs.csv_export1 import conftest

test_header = "Realization,Iteration,Date,A,AZIM_IND_HEIMDAL,CORR_SEIS_HEIMDAL,FLUID_PARAMS:SGCR,FLUID_PARAMS:SWL,H,"\
                "SEIS_COND_HEIMDAL,VARIO_NORM_HEIMDAL,VARIO_PARAL_HEIMDAL,VARIO_VERT_HEIMDAL,VOL_FRAC_HEIMDAL,X,FOPR\n"


design_matrix = "DesignMatrix:" + "DesignMatrix.txt"
dateInterval = "DateInterval:1d"

@pytest.mark.skipif(conftest.find_available_test_data() is None, reason="no ert-statoil test-data")
@pytest.mark.usefixtures("ert_statoil_test_data")
def test_csvexport1_failed_realization_no_summary_file(ert_statoil_test_data):
    path_file = "pathfile_path_to_failed_realizations.txt"
    export_file = "csv_export/EXPORT.txt"
    csv_export1.export(path_file, export_file, design_matrix, "FOPR")
    verifyExportedFile(export_file, test_header, {(0,1), (1,1), (2,1), (2,2)})


@pytest.mark.skipif(conftest.find_available_test_data() is None, reason="no ert-statoil test-data")
@pytest.mark.usefixtures("ert_statoil_test_data")
def test_csvexport1_1_iteration(ert_statoil_test_data):
    path_file = "pathfile.txt"
    export_file = "csv_export/EXPORT.txt"
    csv_export1.export(path_file, export_file, design_matrix, dateInterval, "FOPR")
    verifyExportedFile(export_file, test_header, {(0,1), (1,1), (2,1)})

@pytest.mark.skipif(conftest.find_available_test_data() is None, reason="no ert-statoil test-data")
@pytest.mark.usefixtures("ert_statoil_test_data")
def test_csvexport1_missing_realization(ert_statoil_test_data):
    path_file = "pathfile2.txt"
    export_file = "csv_export/EXPORT.txt"
    csv_export1.export(path_file, export_file, design_matrix, "FOPR")
    verifyExportedFile(export_file, test_header, {(0,1), (2,1)})

@pytest.mark.skipif(conftest.find_available_test_data() is None, reason="no ert-statoil test-data")
@pytest.mark.usefixtures("ert_statoil_test_data")
def test_csvexport1_iterations(ert_statoil_test_data):
    path_file = "pathfile3.txt"
    export_file = "csv_export/EXPORT3.txt"
    csv_export1.export(path_file, export_file, design_matrix, "FOPR")
    verifyExportedFile(export_file, test_header, {(0,1), (1,1), (2,1), (0,2), (1,2), (2,2)})

@pytest.mark.skipif(conftest.find_available_test_data() is None, reason="no ert-statoil test-data")
@pytest.mark.usefixtures("ert_statoil_test_data")
def test_csvexport1_no_iterations(ert_statoil_test_data):
    path_file = "pathfile1.txt"
    export_file = "csv_export/EXPORT.txt"
    csv_export1.export(path_file, export_file, design_matrix, "FOPR")
    verifyExportedFile(export_file, test_header, {(0,0), (1,0), (2,0)})

def verifyExportedFile(exported_file_name, result_header, result_iter_rel):
    with open(exported_file_name, "r") as exported_file:
        lines = exported_file.readlines()

        assert lines[0] == result_header

        iter_rel_no = set()

        for index, line in enumerate(lines[1:]):
            tokens = line.split(",")
            iteration, rel = int(tokens[0]), int(tokens[1])

            if (iteration, rel) not in iter_rel_no:
                iter_rel_no.add((iteration, rel))


        assert iter_rel_no == result_iter_rel

