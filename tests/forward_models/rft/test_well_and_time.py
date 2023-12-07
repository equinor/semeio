import argparse
import datetime

import pytest

from semeio.forward_models.scripts.gendata_rft import load_and_parse_well_time_file


@pytest.fixture()
def initdir(tmpdir):
    tmpdir.chdir()
    valid_data = """
WELL-NAME1 2000-01-01 2
WELL-NAME2 2001-03-02 3
"""
    deprecated_data = """
WELL-NAME1 1 1 2000 2
WELL-NAME2 2 3 2001 3
"""
    tmpdir.join("valid_well_and_time.txt").write(valid_data)
    tmpdir.join("deprecated_well_and_time.txt").write(deprecated_data)
    comment = "-- this is a comment"
    valid_data = comment + "\n" + valid_data + comment
    tmpdir.join("valid_well_and_time_with_comments.txt").write(valid_data)

    incorrect_number = """
WELL-NAME1 2
"""
    tmpdir.join("incorrect_number.txt").write(incorrect_number)

    incorrect_report_format = """
WELL-NAME1 1 1 2000 incorrect_report_step
"""
    tmpdir.join("incorrect_report_format.txt").write(incorrect_report_format)

    incorrect_date_format_1 = """
WELL-NAME1 2013-13-99 1
"""
    tmpdir.join("incorrect_date_format_1.txt").write(incorrect_date_format_1)
    incorrect_date_format_2 = """
WELL-NAME1 day month year 1
"""
    tmpdir.join("incorrect_date_format_2.txt").write(incorrect_date_format_2)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.usefixtures("initdir")
def test_load():
    expected_results = [
        ("WELL-NAME1", datetime.date(2000, 1, 1), 2),
        ("WELL-NAME2", datetime.date(2001, 3, 2), 3),
    ]
    for fname in [
        "valid_well_and_time.txt",
        "deprecated_well_and_time.txt",
        "valid_well_and_time_with_comments.txt",
    ]:
        well_times = load_and_parse_well_time_file(fname)

        for (exp_wname, exp_wtime, exp_report), (wname, wtime, report) in zip(
            expected_results, well_times
        ):
            assert wname == exp_wname
            assert wtime == exp_wtime
            assert report == exp_report


@pytest.mark.usefixtures("initdir")
def test_load_deprecated_date():
    with pytest.warns(FutureWarning, match="YYYY-MM-DD"):
        load_and_parse_well_time_file("deprecated_well_and_time.txt")


@pytest.mark.usefixtures("initdir")
def test_invalid_load():
    errors = [
        "Unexpected number of tokens: expected 3 or 5 (deprecated), got 2",
        "Unable to convert incorrect_report_step to int",
        "Unable to parse date from the line:",
        "Unable to parse date from the line:",
        "The path non_existing does not exist",
    ]

    fnames = [
        "incorrect_number.txt",
        "incorrect_report_format.txt",
        "incorrect_date_format_1.txt",
        "incorrect_date_format_2.txt",
        "non_existing",
    ]

    for fname, error in zip(fnames, errors):
        with pytest.raises(argparse.ArgumentTypeError) as msgcontext:
            load_and_parse_well_time_file(fname)
        assert error in msgcontext.value.args[0]
