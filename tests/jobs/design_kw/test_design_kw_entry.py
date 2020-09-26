import logging
import os
import sys
from distutils.dir_util import copy_tree

import pytest

if sys.version_info.major >= 3:
    from semeio.jobs.scripts import design_kw


@pytest.fixture
def input_data(tmpdir):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    copy_tree(data_dir, tmpdir.strpath)

    cwd = os.getcwd()
    tmpdir.chdir()

    yield

    os.chdir(cwd)


_templatefile = "template.yml.tmpl"
_resultfile = "result.txt"
_log_level = "DEBUG"
_default_log_level = "WARNING"


def test_argparse(input_data):
    args = [_templatefile, _resultfile]
    parser = design_kw.create_parser()
    res = parser.parse_args(args)

    assert res
    assert res.templatefile == _templatefile
    assert res.resultfile == _resultfile
    assert res.log_level == logging.getLevelName(_default_log_level)


def test_argparse_with_logging(input_data):
    args = [_templatefile, _resultfile, "--log-level", _log_level]
    parser = design_kw.create_parser()
    res = parser.parse_args(args)
    assert res
    assert res.templatefile == _templatefile
    assert res.resultfile == _resultfile
    assert res.log_level == logging.getLevelName(_log_level)


def test_argparse_file_not_exists(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["script_name", "file_not_existing.yml.tmpl", _resultfile]
    )
    with pytest.raises(SystemExit):
        design_kw.main_entry_point()


def test_argparse_result_file_missing(input_data, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["script_name", _templatefile])
    with pytest.raises(SystemExit):
        design_kw.main_entry_point()
