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
    data = [
        "key1 14",
        "key2 24",
        "key3 34",
    ]
    assert design_kw.extract_key_value(data) == {
        "key1": "14",
        "key2": "24",
        "key3": "34",
    }


def test_extract_value_with_space():
    data = [
        'key4 "foo bar com"',
        'key5 "conserve   spaces"',
    ]
    assert design_kw.extract_key_value(data) == {
        "key4": "foo bar com",
        "key5": "conserve   spaces",
    }


def test_extract_key_with_space():
    data = ['"key x" "value with space"']
    assert design_kw.extract_key_value(data) == {"key x": "value with space"}


def test_extract_too_many_values():
    """Test that user friendly error message is displayed
    when an error occurs"""

    data = ["SENTENCE this is a string"]
    with pytest.raises(ValueError, match="SENTENCE"):
        design_kw.extract_key_value(data)
    with pytest.raises(ValueError, match="Too many values found"):
        design_kw.extract_key_value(data)


def test_unsupported_quoting():
    data = ['Unclosed " quote']
    with pytest.raises(ValueError, match="No closing quotation"):
        design_kw.extract_key_value(data)


def test_extract_missing_value():
    data = ["key1"]
    with pytest.raises(ValueError, match="No value found"):
        design_kw.extract_key_value(data)


def test_extract_key_value_dup_key():
    data = ["key1 14", "key2 24", "key1 34"]
    with pytest.raises(ValueError, match="multiple times"):
        design_kw.extract_key_value(data)


@pytest.mark.parametrize(
    "paramsdict, expected",
    [
        ({"foo:bar": "value"}, {"bar": "value"}),
        ({"foo:bar": "value", "foo:com": "value2"}, {"bar": "value", "com": "value2"}),
        ({}, {}),
    ],
)
def test_rm_genkw_prefix(paramsdict, expected):
    assert design_kw.rm_genkw_prefix(paramsdict) == expected


@pytest.mark.parametrize(
    "paramsdict, warntext",
    [
        (
            {"foo:bar": "value1", "com:bar": "value2"},
            "Key(s) ['bar'] can only be used with prefix",
        ),
        (
            {"foo:bar": "value1", "bar": "value2"},
            "Key(s) ['bar'] can only be used with prefix",
        ),
    ],
)
def test_rm_genkw_prefix_warnings(paramsdict, warntext, caplog):
    assert design_kw.rm_genkw_prefix(paramsdict) == {}
    assert warntext in caplog.text


@pytest.mark.parametrize(
    "paramsdict, ignores, expected",
    [
        (
            {"LOG10_foo:bar": "value1", "com:bar": "value2"},
            ["LOG10"],
            {"bar": "value2"},
        ),
        ({"LOG10_foo:bar": "value1", "com:bar": "value2"}, ["LOG"], {"bar": "value2"}),
        ({"LOG10PARAM": "value"}, ["LOG10"], {"LOG10PARAM": "value"}),
        ({"foo": "value"}, [""], {"foo": "value"}),
        ({"foo:bar": "value"}, [""], {"bar": "value"}),
    ],
)
def test_rm_genkw_prefix_ignore(paramsdict, ignores, expected):
    assert design_kw.rm_genkw_prefix(paramsdict, ignores) == expected


@pytest.mark.parametrize(
    "paramlines, template, result",
    [
        ("foo:bar 12", "<foo:bar>", "12"),
        ("foo:bar 12", "<bar>", "12"),
        ("foo:bar 12\ncom:bar 13", "<bar>", "<bar>"),  # Yields logged error
        ("LOG10_FOO:foo 1\nFOO:foo 10", "<foo>", "10"),
        ("LOG30_FOO:foo 0.3\nFOO:foo 10", "<foo>", "<foo>"),  # Error logged
        ("foo:bar 12\nbar 12", "<bar>", "12"),
        ("foo:bar 13\nbar 12", "<foo:bar>", "13"),
        ("foo:bar 13\ncom:bar 14", "<bar>", "<bar>"),
        ("foo:bar 13\ncom:bar 14", "<foo:bar>", "13"),
        ("foo:bar 13\ncom:bar 14", "<com:bar>", "14"),
    ],
)
def test_genkw_prefix_handling(paramlines, template, result, tmpdir):
    tmpdir.chdir()
    with open("parameters.txt", "w") as file_h:
        file_h.write(paramlines)
    with open("template.tmpl", "w") as file_h:
        file_h.write(template)
    design_kw.run("template.tmpl", "result.txt", logging.DEBUG)
    with open("result.txt", "r") as file_h:
        resulttxt = "\n".join(file_h.readlines())
    assert resulttxt == result


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


@pytest.mark.parametrize(
    "filenames",
    (
        ["template.yml.reference", "template.yml.tmpl", "template.yml"],
        ["schfile.inc.reference", "schfile.inc.tmpl", "schfile.inc"],
    ),
)
def test_run(input_data, filenames):
    reference_filename = filenames[0]
    template_filename = filenames[1]
    result_filename = filenames[2]

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
    with pytest.raises(ValueError, match="multiple"):
        design_kw.run(
            template_file_name=template_filename,
            result_file_name=result_filename,
            log_level=logging.DEBUG,
            parameters_file_name="parameters_w_duplicates.txt",
        )

    assert not os.path.isfile(design_kw._STATUS_FILE_NAME)
