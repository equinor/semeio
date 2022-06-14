import logging
import os
import shutil
import sys

import pytest

from semeio.jobs.scripts import design_kw


@pytest.fixture
def input_data(tmpdir):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    shutil.copytree(data_dir, tmpdir.strpath, dirs_exist_ok=True)

    cwd = os.getcwd()
    tmpdir.chdir()

    yield

    os.chdir(cwd)


_templatefile = "template.yml.tmpl"
_resultfile = "result.txt"
_log_level = "DEBUG"
_default_log_level = "WARNING"


@pytest.mark.usefixtures("input_data")
def test_argparse():
    args = [_templatefile, _resultfile]
    parser = design_kw.create_parser()
    res = parser.parse_args(args)

    assert res
    assert res.templatefile == _templatefile
    assert res.resultfile == _resultfile
    assert res.log_level == logging.getLevelName(_default_log_level)


@pytest.mark.usefixtures("input_data")
def test_argparse_with_logging():
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


@pytest.mark.usefixtures("input_data")
def test_argparse_result_file_missing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["script_name", _templatefile])
    with pytest.raises(SystemExit):
        design_kw.main_entry_point()
