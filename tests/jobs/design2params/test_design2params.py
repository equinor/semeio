import filecmp
import logging
import os
from distutils.dir_util import copy_tree

import pytest

from semeio.jobs.design2params import design2params


@pytest.fixture
def input_data(tmpdir):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    copy_tree(data_dir, tmpdir.strpath)

    cwd = os.getcwd()
    tmpdir.chdir()

    yield

    os.chdir(cwd)


@pytest.mark.usefixtures("input_data")
@pytest.mark.parametrize(
    "test_file, expected_file",
    [
        (design2params._PARAMETERS_TXT, "refparameters.txt"),
        (design2params._DESIGN_MATRIX_TXT, "refdesignmatrix.txt"),
        (design2params._DESIGN_PARAMETERS_TXT, "refdesignparameters.txt"),
    ],
)
def test_run(test_file, expected_file):
    design2params.run(
        1,
        "design_matrix.xlsx",
        "DesignSheet01",
        "DefaultValues",
        design2params._PARAMETERS_TXT,
        log_level=logging.DEBUG,
    )

    assert filecmp.cmp(test_file, expected_file)

    with open(design2params._TARGET_FILE_TXT, "r") as status_file:
        status = status_file.read()
        assert status == "OK\n"


@pytest.mark.usefixtures("input_data")
@pytest.mark.parametrize(
    "test_file, expected_file",
    [
        ("new_parameters.txt", "refparameters_when_missing.txt"),
        (design2params._DESIGN_MATRIX_TXT, "refdesignmatrix.txt"),
        (design2params._DESIGN_PARAMETERS_TXT, "refparameters_when_missing.txt"),
    ],
)
def test_run_with_no_parameters_txt(test_file, expected_file):
    design2params.run(
        1,
        "design_matrix.xlsx",
        "DesignSheet01",
        "DefaultValues",
        "new_parameters.txt",
        log_level=logging.DEBUG,
    )

    assert filecmp.cmp(test_file, expected_file)

    with open(design2params._TARGET_FILE_TXT, "r") as status_file:
        status = status_file.read()
        assert status == "OK\n"


@pytest.mark.usefixtures("input_data")
@pytest.mark.parametrize(
    "test_file, expected_file",
    [
        (design2params._PARAMETERS_TXT, "refparameters_w_default.txt"),
        (design2params._DESIGN_MATRIX_TXT, "refdesignmatrix.txt"),
        (design2params._DESIGN_PARAMETERS_TXT, "refdesignparameters_w_default.txt"),
    ],
)
def test_run_with_default(test_file, expected_file):
    design2params.run(
        1,
        "design_matrix_tampered_defaults.xlsx",
        "DesignSheet01",
        "DefaultValues",
        design2params._PARAMETERS_TXT,
        log_level=logging.DEBUG,
    )

    assert filecmp.cmp(test_file, expected_file)

    with open(design2params._TARGET_FILE_TXT, "r") as status_file:
        status = status_file.read()
        assert status == "OK\n"


@pytest.mark.usefixtures("input_data")
@pytest.mark.parametrize("realization_id", [(0), (1), (15), (24)])
def test_runs_different_reals_all_ok(realization_id):
    design2params.run(
        realization_id,
        "design_matrix_tampered_defaults.xlsx",
        "DesignSheet01",
        "DefaultValues",
        design2params._PARAMETERS_TXT,
        log_level=logging.DEBUG,
    )

    with open(design2params._TARGET_FILE_TXT, "r") as status_file:
        status = status_file.read()
        assert status == "OK\n"


def test_run_realization_not_exist(input_data):
    with pytest.raises(SystemExit):
        design2params.run(
            1000,
            "design_matrix.xlsx",
            "DesignSheet01",
            "DefaultValues",
            design2params._PARAMETERS_TXT,
            log_level=logging.DEBUG,
        )


def test_open_excel_file_not_existing(input_data):
    with pytest.raises(SystemExit):
        design2params._read_excel("not_existing.xlsx", "DesignSheet01")


def test_open_excel_file_not_xls(input_data):
    with pytest.raises(SystemExit):
        design2params._read_excel("parameters.txt", "DesignSheet01")


def test_open_excel_file_header_missing(input_data):
    with pytest.raises(SystemExit):
        design2params.run(
            1,
            "design_matrix_missing_header.xlsx",
            "DesignSheet01",
            "DefaultValues",
            design2params._PARAMETERS_TXT,
            log_level=logging.DEBUG,
        )


def test_open_excel_file_value_missing(input_data):
    with pytest.raises(SystemExit):
        design2params.run(
            1,
            "design_matrix_missing_value.xlsx",
            "DesignSheet01",
            "DefaultValues",
            design2params._PARAMETERS_TXT,
            log_level=logging.DEBUG,
        )
