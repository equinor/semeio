import logging
import os
import sys
from distutils.dir_util import copy_tree

import pytest

if sys.version_info.major >= 3:
    from semeio.jobs.scripts import design2params


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
@pytest.fixture
def input_data(tmpdir):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    copy_tree(data_dir, tmpdir.strpath)

    cwd = os.getcwd()
    tmpdir.chdir()

    yield

    os.chdir(cwd)


_realization = 1
_xls = "design_matrix.xlsx"
_sheet = "DesignSheet01"
_default_sheet = "DefaultSheet01"
_parametersfilename = "parameters.txt"
_log_level = "DEBUG"
_default_log_level = "WARNING"


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
def test_argparse(input_data):
    args = [str(_realization), _xls, _sheet, _default_sheet]
    parser = design2params.create_parser()
    res = parser.parse_args(args)
    assert res
    assert res.realization == _realization
    assert res.xlsfilename == _xls
    assert res.designsheetname == _sheet
    assert res.defaultssheetname == _default_sheet


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
def test_argparse_no_default(input_data):
    args = [str(_realization), _xls, _sheet]
    parser = design2params.create_parser()
    res = parser.parse_args(args)
    assert res
    assert res.realization == _realization
    assert res.xlsfilename == _xls
    assert res.designsheetname == _sheet
    assert res.defaultssheetname == None


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
def test_argparse_with_optionals(input_data):
    args = [
        str(_realization),
        _xls,
        _sheet,
        "-p",
        _parametersfilename,
        "-l",
        _log_level,
    ]
    parser = design2params.create_parser()
    res = parser.parse_args(args)
    assert res
    assert res.realization == _realization
    assert res.xlsfilename == _xls
    assert res.designsheetname == _sheet
    assert res.defaultssheetname == None
    assert res.parametersfilename == _parametersfilename
    assert res.log_level == logging.getLevelName(_log_level)


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
def test_argparse_xls_file_not_exists():
    with pytest.raises(SystemExit):
        design2params.main([str(_realization), "not_a_file", _sheet])


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
def test_argparse_parameters_file_not_exists(input_data):
    not_existing_parameters_txt = "not_a_file.txt"

    args = [
        str(_realization),
        _xls,
        _sheet,
        "-p",
        not_existing_parameters_txt,
    ]
    parser = design2params.create_parser()
    res = parser.parse_args(args)
    assert res
    assert res.realization == _realization
    assert res.xlsfilename == _xls
    assert res.designsheetname == _sheet
    assert res.defaultssheetname == None
    assert res.parametersfilename == not_existing_parameters_txt
    assert res.log_level == logging.getLevelName(_default_log_level)


@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3")
def test_argparse_design_sheet_missing(input_data):
    with pytest.raises(SystemExit):
        design2params.main([str(_realization), _xls])
