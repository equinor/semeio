import logging
import os
import shutil
import sys

import pytest

from semeio.forward_models.scripts import design2params


@pytest.fixture
def input_data(tmpdir):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    shutil.copytree(data_dir, tmpdir.strpath, dirs_exist_ok=True)

    cwd = os.getcwd()
    tmpdir.chdir()

    yield

    os.chdir(cwd)


_REALIZATION = 1
_XLS = "design_matrix.xlsx"
_SHEET = "DesignSheet01"
_DEFAULT_SHEET = "DefaultSheet01"
_PARAMETERSFILENAME = "parameters.txt"
_LOG_LEVEL = "DEBUG"
_DEFAULT_LOG_LEVEL = "WARNING"


@pytest.mark.usefixtures("input_data")
def test_argparse():
    args = [str(_REALIZATION), _XLS, _SHEET, _DEFAULT_SHEET]
    parser = design2params.create_parser()
    res = parser.parse_args(args)
    assert res
    assert res.realization == _REALIZATION
    assert res.xlsfilename == _XLS
    assert res.designsheetname == _SHEET
    assert res.defaultssheetname == _DEFAULT_SHEET


@pytest.mark.usefixtures("input_data")
def test_argparse_no_default():
    args = [str(_REALIZATION), _XLS, _SHEET]
    parser = design2params.create_parser()
    res = parser.parse_args(args)
    assert res
    assert res.realization == _REALIZATION
    assert res.xlsfilename == _XLS
    assert res.designsheetname == _SHEET
    assert res.defaultssheetname is None


@pytest.mark.usefixtures("input_data")
def test_argparse_with_optionals():
    args = [
        str(_REALIZATION),
        _XLS,
        _SHEET,
        "-p",
        _PARAMETERSFILENAME,
        "-l",
        _LOG_LEVEL,
    ]
    parser = design2params.create_parser()
    res = parser.parse_args(args)
    assert res
    assert res.realization == _REALIZATION
    assert res.xlsfilename == _XLS
    assert res.designsheetname == _SHEET
    assert res.defaultssheetname is None
    assert res.parametersfilename == _PARAMETERSFILENAME
    assert res.log_level == logging.getLevelName(_LOG_LEVEL)


def test_argparse_xls_file_not_exists(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["path", str(_REALIZATION), "not_a_file", _SHEET])
    with pytest.raises(SystemExit):
        design2params.main_entry_point()


@pytest.mark.usefixtures("input_data")
def test_argparse_parameters_file_not_exists():
    not_existing_parameters_txt = "not_a_file.txt"

    args = [
        str(_REALIZATION),
        _XLS,
        _SHEET,
        "-p",
        not_existing_parameters_txt,
    ]
    parser = design2params.create_parser()
    res = parser.parse_args(args)
    assert res
    assert res.realization == _REALIZATION
    assert res.xlsfilename == _XLS
    assert res.designsheetname == _SHEET
    assert res.defaultssheetname is None
    assert res.parametersfilename == not_existing_parameters_txt
    assert res.log_level == logging.getLevelName(_DEFAULT_LOG_LEVEL)


@pytest.mark.usefixtures("input_data")
def test_argparse_design_sheet_missing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["path", str(_REALIZATION), _XLS])
    with pytest.raises(SystemExit):
        design2params.main_entry_point()
