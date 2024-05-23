import filecmp
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from semeio._exceptions.exceptions import ValidationError
from semeio.forward_models.design2params import design2params
from semeio.forward_models.design_kw.design_kw import extract_key_value


@pytest.fixture
def input_data(tmpdir):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    shutil.copytree(data_dir, tmpdir.strpath, dirs_exist_ok=True)

    cwd = os.getcwd()
    tmpdir.chdir()

    yield

    os.chdir(cwd)


# pylint: disable=protected-access
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


@pytest.mark.usefixtures("input_data")
@pytest.mark.parametrize(
    "test_file, expected_file",
    [
        (design2params._PARAMETERS_TXT, "refparameters_w_spaces.txt"),
        (design2params._DESIGN_MATRIX_TXT, "refdesignmatrix_w_spaces.txt"),
        (design2params._DESIGN_PARAMETERS_TXT, "refdesignparameters_w_spaces.txt"),
    ],
)
def test_run_with_spaces_in_cells(test_file, expected_file):
    design2params.run(
        1,
        "design_matrix_cells_with_spaces.xlsx",
        "DesignSheet01",
        "DefaultValues",
        design2params._PARAMETERS_TXT,
        log_level=logging.DEBUG,
    )

    assert filecmp.cmp(test_file, expected_file)


@pytest.mark.usefixtures("input_data")
@pytest.mark.parametrize("realization_id", [(0), (1), (15), (24)])
def test_runs_different_reals_no_raise(realization_id):
    design2params.run(
        realization_id,
        "design_matrix_tampered_defaults.xlsx",
        parametersfilename=design2params._PARAMETERS_TXT,
        log_level=logging.DEBUG,
    )


def test_empty_defaults(tmpdir):
    tmpdir.chdir()
    write_design_xlsx("design_matrix.xlsx")

    parsed_defaults = design2params._read_defaultssheet(
        "design_matrix.xlsx", "DefaultValues"
    )
    assert parsed_defaults.empty
    assert (parsed_defaults.columns == ["keys", "defaults"]).all()


def test_one_column_defaults(tmpdir):
    tmpdir.chdir()
    write_design_xlsx("design_matrix.xlsx", defaultsdf=pd.DataFrame(data=[["foo"]]))

    with pytest.raises(SystemExit):
        design2params._read_defaultssheet("design_matrix.xlsx", "DefaultValues")


def test_three_column_defaults(tmpdir):
    """Anything other than the two first columns is ignored"""
    tmpdir.chdir()
    write_design_xlsx(
        "design_matrix.xlsx", defaultsdf=pd.DataFrame(data=[["foo", "bar", "com"]])
    )

    parsed_defaults = design2params._read_defaultssheet(
        "design_matrix.xlsx", "DefaultValues"
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame(columns=["keys", "defaults"], data=[["foo", "bar"]]),
        parsed_defaults,
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("ext, engine", [("ods", "odf"), ("xlsx", None)])
def test_support_multiple_formats(tmp_path, ext, engine):
    fname = tmp_path / ("test." + ext)
    dframe = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    dframe.to_excel(fname, sheet_name="TEST", index=False, engine=engine)
    loaded_df = design2params._read_excel(
        fname,
        sheet_name="TEST",
    )
    loaded_df["col1"] = pd.to_numeric(loaded_df["col1"])
    loaded_df["col2"] = pd.to_numeric(loaded_df["col2"])
    pd.testing.assert_frame_equal(dframe, loaded_df, check_dtype=False)


@pytest.mark.usefixtures("input_data")
def test_run_realization_not_exist():
    assert Path("design_matrix.xlsx").exists()
    with pytest.raises(SystemExit):
        design2params.run(
            1000,
            "design_matrix.xlsx",
            "DesignSheet01",
            "DefaultValues",
            design2params._PARAMETERS_TXT,
            log_level=logging.DEBUG,
        )


@pytest.mark.usefixtures("input_data")
def test_open_excel_file_not_existing():
    assert Path("design_matrix.xlsx").exists()
    assert not Path("not_existing.xlsx").exists()
    with pytest.raises(SystemExit):
        design2params._read_excel("not_existing.xlsx", "DesignSheet01")


@pytest.mark.usefixtures("input_data")
def test_open_excel_file_not_xls():
    assert Path("parameters.txt").exists()
    with pytest.raises(SystemExit):
        design2params._read_excel("parameters.txt", "DesignSheet01")


@pytest.mark.usefixtures("input_data")
def test_open_excel_file_header_missing():
    with pytest.raises(
        ValidationError,
        match="Design matrix not valid, error: Column headers not present in column",
    ):
        design2params.run(
            1,
            "design_matrix_missing_header.xlsx",
            "DesignSheet01",
            "DefaultValues",
            design2params._PARAMETERS_TXT,
            log_level=logging.DEBUG,
        )


@pytest.mark.usefixtures("input_data")
def test_open_excel_file_value_missing(caplog):
    """Check that we get SystemExit only for those realizations
    affected by missing cell values

    (all realizations see warnings for empty cells in other realizations)
    """
    reals_with_missing_values = [7, 11]
    for real in range(0, 25):
        if real in reals_with_missing_values:
            with pytest.raises(SystemExit):
                design2params.run(
                    real,
                    "design_matrix_missing_value.xlsx",
                    "DesignSheet01",
                    "DefaultValues",
                    design2params._PARAMETERS_TXT,
                    log_level=logging.DEBUG,
                )
        else:
            design2params.run(
                real,
                "design_matrix_missing_value.xlsx",
                "DesignSheet01",
                "DefaultValues",
                design2params._PARAMETERS_TXT,
                log_level=logging.DEBUG,
            )
    assert "Realization 7, column RMS_SEED" in caplog.text
    assert "Realization 11, column MULTFLT_F1" in caplog.text


@pytest.mark.parametrize(
    "exist_params, design_m, expected_warning",
    [
        ("NAMESPACE:FOO 3", {"FOO": 5}, None),
        ("FOO 3", {"FOO": 4}, "Parameter FOO already exists"),
        ("FOO 3", {"FOO": 3}, None),  # Silently ignore this situation
        ("", {"FOO": 3}, None),
        ("FOO 4\nBAR 5", {"BAR": 6}, "Parameter BAR already exists"),
        ("FOO 4\nBAR 5", {"FOO": 4}, None),
        ("FOO 4\nBAR 5", {"FOO": 5}, "Parameter FOO already exists"),
        ("FOO 4\nBAR 5", {"FOO": 4, "BAR": 6}, "Parameter BAR already exists"),
    ],
)
def test_existing_parameterstxt(
    exist_params, design_m, expected_warning, tmpdir, caplog
):
    """Test warnings emitted when the file parameters.txt is prefilled"""
    tmpdir.chdir()

    params_file = "parameters.txt"
    with open(params_file, "w", encoding="utf-8") as file_h:
        file_h.write(exist_params)

    designsheet_df = pd.DataFrame.from_records(data=[design_m])
    designsheet_df.insert(0, "REAL", [0])

    write_design_xlsx("design_matrix.xlsx", designdf=designsheet_df)

    design2params.run(
        0,
        "design_matrix.xlsx",
        parametersfilename=params_file,
        log_level=logging.DEBUG,
    )

    if expected_warning:
        assert expected_warning in caplog.text
    else:
        assert "already exists" not in caplog.text


@pytest.mark.usefixtures("input_data")
def test_logging_order(tmpdir, caplog):
    """Test that the warnings emitted when parameters.txt is prefilled
    respects the order from parameters.txt. Pandas 2.2 changed the default
    sorting behaviour of merge, and this checks that the sorting works as we expect."""
    tmpdir.chdir()

    params_file = "parameters.txt"
    data_dict = {}
    expected_warning = []
    with open(params_file, encoding="utf-8") as file_h:
        for line in file_h:
            key, value = line.strip().split()
            data_dict[key] = 0
            expected_warning.append(
                f"Parameter {key} already exists in parameters.txt "
                f"with value {float(value)}, design matrix value 0 ignored"
            )
    # We reverse the order of the keys to detect if the logging still happens
    # in the order set by parameters.txt
    designsheet_df = pd.DataFrame.from_records([data_dict]).iloc[:, ::-1]
    designsheet_df.insert(0, "REAL", [0])

    write_design_xlsx("design_matrix.xlsx", designdf=designsheet_df)

    design2params.run(
        0,
        "design_matrix.xlsx",
        parametersfilename=params_file,
        log_level=logging.DEBUG,
    )
    actual_warnings = [
        record.message for record in caplog.records if record.levelname == "WARNING"
    ]

    assert actual_warnings == expected_warning


@pytest.mark.usefixtures("input_data")
def test_open_excel_file_wrong_defaults():
    with pytest.raises(SystemExit):
        design2params.run(
            1,
            "design_matrix_missing_value.xlsx",
            "DesignSheet01",
            "DefaultValuesWRONGNAME",
            design2params._PARAMETERS_TXT,
            log_level=logging.DEBUG,
        )


@pytest.mark.parametrize(
    "cellvalue, expected_parameters_str",
    [
        ("TRUE", "TRUE"),
        ("FALSE", "FALSE"),
        ("True", "True"),
        (True, "True"),
        (False, "False"),
        (0, "0"),
        (0.000, "0"),
        (1, "1"),
        ("1", "1"),
        ("1.0", "1.0"),
        (1.0, "1"),  # This difference is related to to_excel()
        ("æøå", "æøå"),
        (1e-1, "0.1"),
        (1e-4, "0.0001"),
        (1e-5, "1e-05"),
        (1e-16, "1e-16"),
        (1e3, "1000"),
        (1e10, "10000000000"),
    ],
)
def test_single_cell_values(cellvalue, expected_parameters_str, tmpdir):
    """Test how certain single values go through the Excel input sheets
    and all the way to parameters.txt"""
    tmpdir.chdir()
    write_design_xlsx(
        "design_matrix.xlsx",
        designdf=pd.DataFrame(columns=["REAL", "SOMEKEY"], data=[[0, cellvalue]]),
    )

    params_file = "parameters.txt"
    design2params.run(
        0,
        "design_matrix.xlsx",
        parametersfilename=params_file,
        log_level=logging.DEBUG,
    )
    with open(params_file, encoding="utf-8") as p_file:
        params_lines = p_file.readlines()
    key_vals = extract_key_value(params_lines)
    assert key_vals["SOMEKEY"] == expected_parameters_str


@pytest.mark.parametrize(
    "cellvalues, expected_parameters_strs",
    [
        (["TRUE", "True"], ["TRUE", "True"]),
        (["TRUE", 1], ["TRUE", "1"]),
        (["True", 1.0], ["True", "1"]),
        (["True", "1.0"], ["True", "1.0"]),
        ([0, "1.0"], ["0", "1.0"]),
        ([1, 1.2], ["1", "1.2"]),
        (["æ", 1e10], ["æ", "10000000000"]),
    ],
)
def test_pair_cell_values(cellvalues, expected_parameters_strs, tmpdir):
    """Test how a pair of values, one for each realization, go through
    the Excel input sheets and all the way to parameters.txt.

    This is to ensure that differing datatypes in a single Excel columns does
    not affect individual values."""
    tmpdir.chdir()

    write_design_xlsx(
        "design_matrix.xlsx",
        designdf=pd.DataFrame(
            columns=["REAL", "SOMEKEY"], data=[[0, cellvalues[0]], [1, cellvalues[1]]]
        ),
    )

    params_0 = "parameters0.txt"
    design2params.run(
        0,
        "design_matrix.xlsx",
        parametersfilename=params_0,
        log_level=logging.DEBUG,
    )
    params_1 = "parameters1.txt"
    design2params.run(
        1,
        "design_matrix.xlsx",
        parametersfilename=params_1,
        log_level=logging.DEBUG,
    )
    with open(params_0, encoding="utf-8") as p_file:
        params_lines_0 = p_file.readlines()
    with open(params_1, encoding="utf-8") as p_file:
        params_lines_1 = p_file.readlines()
    key_vals_real0 = extract_key_value(params_lines_0)
    key_vals_real1 = extract_key_value(params_lines_1)
    assert key_vals_real0["SOMEKEY"] == expected_parameters_strs[0]
    assert key_vals_real1["SOMEKEY"] == expected_parameters_strs[1]


@pytest.mark.parametrize(
    "paramnames",
    (
        [
            ["REAL", "SOMEKEY "],
            ["REAL", " SOMEKEY"],
            ["REAL", " SOMEKEY "],
            ["REAL", "\tSOMEKEY"],
            ["REAL", "SOMEKEY\t"],
            ["REAL", "SOMEKEY\n"],
        ]
    ),
)
def test_headers_trailing_whitespace(paramnames, tmpdir):
    """Parameter names can have un-noticed trailing whitespace in Excel.

    This can happen both in the design- and the defaultssheet.

    This should error hard as it has no believed use-case and only
    creates user confusion.
    """
    tmpdir.chdir()

    write_design_xlsx(
        "design_matrix.xlsx",
        designdf=pd.DataFrame(columns=paramnames, data=[[0, "foo"]]),
    )
    write_design_xlsx(
        "design_matrix_onlydefaults.xlsx",
        defaultsdf=pd.DataFrame(data=[[paramnames[1], "bar"]]),
    )

    with pytest.raises(SystemExit, match="whitespace"):
        design2params.run(
            0,
            "design_matrix.xlsx",
        )
    with pytest.raises(SystemExit, match="whitespace"):
        design2params.run(
            0,
            "design_matrix_onlydefaults.xlsx",
            parametersfilename="parameters_onlydefaults.txt",
        )


@pytest.mark.parametrize("paramname", design2params.DENYLIST)
def test_denylist_designsheet(paramname, tmpdir):
    """Some parameter names are reserved, ensure we error hard if found."""
    tmpdir.chdir()

    write_design_xlsx(
        "design_matrix.xlsx",
        designdf=pd.DataFrame(columns=["REAL", paramname], data=[[0, "foo"]]),
    )
    with pytest.raises(SystemExit, match="not allowed"):
        design2params.run(
            0,
            "design_matrix.xlsx",
        )


@pytest.mark.parametrize("paramname", design2params.DENYLIST_DEFAULTS)
def test_denylist_defaults(paramname, tmpdir):
    """Some parameter names are reserved, ensure we error hard if found.

    The defaults sheet can have more denied parameters than the designsheet,
    in particular the "REAL" column."""
    tmpdir.chdir()
    write_design_xlsx(
        "design_matrix.xlsx", defaultsdf=pd.DataFrame(data=[[paramname, "bar"]])
    )
    with pytest.raises(SystemExit, match="not allowed"):
        design2params.run(
            0,
            "design_matrix.xlsx",
        )


@pytest.mark.parametrize("paramname", [1, 1.0])
def test_invalid_pandas_header_error(tmpdir, paramname):
    tmpdir.chdir()

    write_design_xlsx(
        "design_matrix.xlsx",
        designdf=pd.DataFrame(columns=["REAL", paramname], data=[[0, "foo"]]),
    )

    expected_error = "Invalid value in design matrix header"
    with pytest.raises(ValidationError, match=expected_error):
        design2params.run(
            0,
            "design_matrix.xlsx",
        )


@pytest.mark.parametrize("paramname", [1, 1.0])
def test_denylist_expected_error(paramname, tmpdir):
    tmpdir.chdir()
    write_design_xlsx(
        "design_matrix.xlsx", defaultsdf=pd.DataFrame(data=[[paramname, "bar"]])
    )
    with pytest.raises(SystemExit, match="does not exist in design matrix"):
        design2params.run(
            0,
            "design_matrix.xlsx",
        )


@pytest.mark.parametrize(
    "paramset", [["REAL", "REAL"], ["REAL", "REAL", "REAL"], ["FOO", "FOO"]]
)
def test_duplicated_designcolumns(paramset, tmpdir, caplog):
    """If the excel sheet contains duplicated column headers, pandas will
    happily read it but modify column names. This is most likely a user error,
    and a warning is issued"""

    tmpdir.chdir()
    write_design_xlsx(
        "design_matrix.xlsx",
        designdf=pd.DataFrame(
            columns=["REAL"] + paramset, data=[[0] + ["foo"] * len(paramset)]
        ),
    )

    design2params.run(
        0,
        "design_matrix.xlsx",
    )
    assert "are probably duplicated" in caplog.text


def test_equal_sheetargs(tmpdir):
    """The dataframes for design sheet and defaults have a different structure
    and cannot be allowed to be mixed"""
    tmpdir.chdir()
    write_design_xlsx(
        "dummy.xlsx",
        designdf=pd.DataFrame(columns=["COM", "FOO"], data=[[0, "BAR"]]),
        designsheetname="foo",
    )
    with pytest.raises(SystemExit):
        design2params.run(
            0, "dummy.xlsx", designsheetname="foo", defaultssheetname="foo"
        )


def write_design_xlsx(
    filename,
    designdf=None,
    defaultsdf=None,
    designsheetname="DesignSheet01",
    defaultssheetname="DefaultValues",
):
    """Generate an XLS file with a design sheet and a defaults sheet.

    Args:
        filename (str)
        designdf (pd.DataFrame): Design data. Empty data will be written if None
        defaultsdf (pd.DataFrame): defaults data. Empty data will be written if None
        designsheetname (str): sheet-name in xlsx file. Defaulted if not provided
        defaultssheetname (str): sheet-name in xlsx file. Defaulted if not provided
    """
    # pylint: disable=abstract-class-instantiated
    with pd.ExcelWriter(filename) as writer:
        if designdf is not None:
            designdf.to_excel(writer, sheet_name=designsheetname, index=False)
        else:
            # Write empty sheet.
            pd.DataFrame().to_excel(writer, sheet_name=designsheetname, index=False)

        if defaultsdf is not None:
            defaultsdf.to_excel(
                writer, sheet_name=defaultssheetname, index=False, header=None
            )
        else:
            # Write empty sheet.
            pd.DataFrame().to_excel(
                writer, sheet_name=defaultssheetname, index=False, header=None
            )
