import logging
import os
import shutil
import sys

import pytest

from semeio.forward_models.scripts import design_kw


@pytest.fixture
def input_data(tmpdir):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    shutil.copytree(data_dir, tmpdir.strpath, dirs_exist_ok=True)

    cwd = os.getcwd()
    tmpdir.chdir()

    yield

    os.chdir(cwd)


_TEMPLATEFILE = "template.yml.tmpl"
_TEMPLATEFILE_UNMATCHED = "template_unmatched.yml.tmpl"
_RESULTFILE = "result.txt"
_LOG_LEVEL = "DEBUG"
_DEFAULT_LOG_LEVEL = "WARNING"


@pytest.mark.usefixtures("input_data")
def test_argparse():
    args = [_TEMPLATEFILE, _RESULTFILE]
    parser = design_kw.create_parser()
    res = parser.parse_args(args)

    assert res
    assert res.templatefile == _TEMPLATEFILE
    assert res.resultfile == _RESULTFILE
    assert res.log_level == logging.getLevelName(_DEFAULT_LOG_LEVEL)


@pytest.mark.usefixtures("input_data")
def test_argparse_with_logging():
    args = [_TEMPLATEFILE, _RESULTFILE, "--log-level", _LOG_LEVEL]
    parser = design_kw.create_parser()
    res = parser.parse_args(args)
    assert res
    assert res.templatefile == _TEMPLATEFILE
    assert res.resultfile == _RESULTFILE
    assert res.log_level == logging.getLevelName(_LOG_LEVEL)


def test_argparse_file_not_exists(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["script_name", "file_not_existing.yml.tmpl", _RESULTFILE]
    )
    with pytest.raises(SystemExit):
        design_kw.main_entry_point()


@pytest.mark.usefixtures("input_data")
def test_argparse_result_file_missing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["script_name", _TEMPLATEFILE])
    with pytest.raises(SystemExit):
        design_kw.main_entry_point()


@pytest.mark.usefixtures("input_data")
def test_unmatched_template(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["script_name", _TEMPLATEFILE_UNMATCHED, _RESULTFILE]
    )
    with pytest.raises(SystemExit):
        design_kw.main_entry_point()
