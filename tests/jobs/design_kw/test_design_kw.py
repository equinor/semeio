import logging
import os
from distutils.dir_util import copy_tree

import pytest

from semeio.jobs.design_kw import design_kw


@pytest.fixture
def input_data(tmpdir):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    copy_tree(data_dir, tmpdir.strpath)

    cwd = os.getcwd()
    tmpdir.chdir()

    yield

    os.chdir(cwd)


def test_extract_key_value_ok():
    data = ["key1 14", "key2 24", "key3 34"]
    assert design_kw.extract_key_value(data) == {
        "key1": "14",
        "key2": "24",
        "key3": "34",
    }


def test_extract_key_value_dup_key():
    data = ["key1 14", "key2 24", "key1 34"]
    with pytest.raises(SystemExit):
        design_kw.extract_key_value(data)


def test_is_comment():
    line = "# this is a std pattern comment"
    assert design_kw.is_comment(line)

    line = "-- this is a ecl comment"
    assert design_kw.is_comment(line)

    line = "  ECLIPSE_INIT_DATE: [2000-01-01]"
    assert not design_kw.is_comment(line)


def test_unmatched_templates():
    line = "  ECLIPSE_INIT_DATE: [2000-01-01]"
    assert not design_kw.unmatched_templates(line)

    line = "  ECLIPSE_INIT_DATE: <DATE>"
    assert design_kw.unmatched_templates(line) == ["<DATE>"]


def test_is_perl():
    file_name = "filename.yml"
    template = ["perl", "  ECLIPSE_INIT_DATE: <DATE>"]
    assert design_kw.is_perl(file_name, template)

    file_name = "filename.pl"
    template = ["perl", "  ECLIPSE_INIT_DATE: <DATE>"]
    assert design_kw.is_perl(file_name, template)

    file_name = "filename.pl"
    template = ["  ECLIPSE_INIT_DATE: <DATE>"]
    assert design_kw.is_perl(file_name, template)

    file_name = "filename.yml"
    template = ["  ECLIPSE_INIT_DATE: <DATE>"]
    assert not design_kw.is_perl(file_name, template)


def test_run(input_data):
    reference_filename = "template.yml.reference"
    template_filename = "template.yml.tmpl"
    result_filename = "template.yml"

    design_kw.run(template_filename, result_filename, log_level=logging.DEBUG)

    with open(result_filename, "r") as result_file:
        result = result_file.read()

    with open(reference_filename, "r") as reference_file:
        reference = reference_file.read()

    with open(design_kw._STATUS_FILE_NAME, "r") as status_file:
        status = status_file.read()

    assert result == reference
    assert status == "DESIGN_KW OK\n"


def test_run_unmatched(input_data):
    template_filename = "template_unmatched.yml.tmpl"
    result_filename = "template.yml"
    design_kw.run(
        template_file_name=template_filename,
        result_file_name=result_filename,
        log_level=logging.DEBUG,
        parameters_file_name="parameters.txt",
    )

    assert not os.path.isfile(design_kw._STATUS_FILE_NAME)


def test_run_duplicate_keys(input_data):
    template_filename = "template.yml.tmpl"
    result_filename = "template.yml"
    with pytest.raises(SystemExit):
        design_kw.run(
            template_file_name=template_filename,
            result_file_name=result_filename,
            log_level=logging.DEBUG,
            parameters_file_name="parameters_w_duplicates.txt",
        )

    assert not os.path.isfile(design_kw._STATUS_FILE_NAME)
