import numpy as np
import pandas as pd

import warnings
import logging

from semeio._exceptions.exceptions import ValidationError

# Filenames created/modified by this script.
# Don't change unless you understand the consequences.
_TARGET_FILE_TXT = "DESIGN2PARAMS.OK"
_DESIGN_MATRIX_TXT = "designmatrix.txt"
_DESIGN_PARAMETERS_TXT = "designparameters.txt"
_PARAMETERS_TXT = "parameters.txt"

# These parameter names are reserved and cannot
# be used in design matrices.
DENYLIST = ["ENSEMBLE", "DATE"]
DENYLIST_DEFAULTS = DENYLIST + ["REAL"]

logger = logging.getLogger(__name__)


def run(
    realization,
    xlsfilename,
    designsheetname="DesignSheet01",
    defaultssheetname="DefaultValues",
    parametersfilename="parameters.txt",
    log_level=None,
):
    """
    Reads out all file content from different files and create dataframes
    """
    if log_level is not None:
        logger.setLevel(log_level)

    design_matrix_sheet = _read_excel(xlsfilename, designsheetname)

    try:
        _validate_design_matrix_header(design_matrix_sheet)
    except ValueError as err:
        raise ValidationError(f"Design matrix not valid, error: {str(err)}") from err

    if realization in _invalid_design_realizations(design_matrix_sheet):
        raise SystemExit("Design parameters invalid for current realization")

    if designsheetname == defaultssheetname:
        raise SystemExit("Design-sheet and defaults-sheet can not be the same")

    default_df = _read_defaultssheet(xlsfilename, defaultssheetname)

    try:
        parameters = pd.read_csv(parametersfilename, delimiter=" ", header=None)
    except IOError:
        logger.info(
            "No {} exists, creating a new, empty one.".format(parametersfilename)
        )
        parameters = pd.DataFrame(columns=[0, 1])
    except pd.errors.EmptyDataError:
        logger.info("{} existed but was empty.".format(parametersfilename))
        parameters = pd.DataFrame(columns=[0, 1])
    parameters.rename(columns={0: "keys", 1: "parameters"}, inplace=True)

    _complete_parameters_file(
        realization, parameters, parametersfilename, design_matrix_sheet, default_df
    )


def _complete_parameters_file(
    realization, parameters, parametersfilename, design_matrix_sheet, default_sheet
):
    """
    Pick key / values from choosen realization in design matrix
    Append those key / values if not present into parameters.txt
    Create file adding those key / values to designparameters.txt
    if default-sheet contains key / values not existing in
    design matrix nor parameters, append those as well to both
    parameters.txt and designparameters.txt
    Write designmatrix.txt as a csv repr of xls

    :raises: SystemExit if design matrix contains empty headers
    :raises: SystemExit design matrix contains empty cells
    :raises: SystemExit if provided realization not exists
    """
    # get header / vals for chosen realization
    try:
        realization_values = pd.DataFrame(design_matrix_sheet.iloc[realization, 1:])
    except IndexError as err:
        raise SystemExit(
            "Provided realization arg {} does not exist in design matrix".format(
                realization
            )
        ) from err
    realization_values.reset_index(inplace=True)
    realization_values.rename(
        columns={"index": "keys", realization: "realization"}, inplace=True
    )

    # merge realization from design-matrix with parameters and defaults to one dataframe
    merged = pd.merge(
        parameters,
        pd.merge(realization_values, default_sheet, on="keys", how="outer"),
        on="keys",
        how="outer",
    )

    # add new column with a combined parameters / data
    # from the realization in design matrix
    merged["parameters_realization"] = merged["parameters"].combine_first(
        merged["realization"]
    )

    # add new column with a total combined of parameters_realization and defaults
    merged["combined"] = merged["parameters_realization"].combine_first(
        merged["defaults"]
    )

    # keys in the design matrix not present in parameters
    design_parameters = merged[
        merged["parameters"].isnull() & merged["realization"].notnull()
    ]
    logger.info(
        "\ndesign parameters used: \n%s", design_parameters[["keys", "combined"]]
    )

    # keys without value in parameters and design matrix
    defaults = merged[merged["parameters_realization"].isnull()]
    logger.info("\ndefaults used: \n%s", defaults[["keys", "combined"]])

    # warn if keys with different values in parameters.tst and design matrix
    conflicts = merged[
        ~(merged["combined"].astype(str) == merged["realization"].astype(str))
        & (~merged["realization"].isnull())
    ]
    if not conflicts.empty:
        for _, row in conflicts.iterrows():
            msg = (
                "Parameter {} already exists in {} with value {}, "
                "design matrix value {} ignored"
            ).format(
                row["keys"], parametersfilename, row["parameters"], row["realization"]
            )
            logger.warning(msg)

    # write used design matrix to csv
    design_matrix_sheet.to_csv(
        _DESIGN_MATRIX_TXT,
        sep=" ",
        index=False,
        float_format="%.10f",
        na_rep="__VOID__",
    )

    # keys not present in parameters
    combined_parameters = merged[merged["parameters"].isnull()]

    # append new keys to parameters.txt
    combined_parameters.to_csv(
        parametersfilename,
        columns=["keys", "combined"],
        sep=" ",
        mode="a",
        header=False,
        index=False,
    )

    # write file with the new keys to designparameters.txt
    combined_parameters.to_csv(
        _DESIGN_PARAMETERS_TXT,
        columns=["keys", "combined"],
        sep=" ",
        header=False,
        index=False,
    )

    # if all ok - write the ok file
    with open(_TARGET_FILE_TXT, "w") as target_file:
        target_file.write("OK\n")


def _read_excel(file_name, sheet_name, header=0):
    """
    Make dataframe from excel file
    :return: Dataframe
    :raises: SystemExit if file not found
    :raises: SystemExit if file not loaded correctly
    """
    try:
        df = pd.read_excel(file_name, sheet_name, header=header, dtype=str)
        return df.dropna(axis=1, how="all")
    except IOError as err:
        raise SystemExit("File {} not found".format(file_name)) from err
    except Exception as err:
        raise SystemExit(
            (
                "File {} is probably not of correct type. Failed with exception '{}'"
            ).format(file_name, str(err))
        ) from err


def _validate_design_matrix_header(design_matrix):
    """
    Validate header in user inputted design matrix
    :raises: ValueError if design matrix contains empty headers
    """
    try:
        unnamed = design_matrix.loc[:, design_matrix.columns.str.contains("^Unnamed")]
    except ValueError as err:
        # We catch because int/floats as column headers
        # in xlsx gets read as int/float and is not valid to index by.
        raise ValueError(
            f"Invalid value in design matrix header, error: {str(err)}"
        ) from err
    column_indexes = [int(x.split(":")[1]) for x in unnamed.columns.values]
    if len(column_indexes) > 0:
        raise ValueError(
            "Column headers not present in column {}".format(column_indexes)
        )


def _invalid_design_realizations(design_matrix):
    """
    Build a set of realization indices where something is wrong,
    f.ex empty cells

    Logs warnings for all empty cells found in all realizations.

    Logs warning if it looks like some column names have been
    repeated in the source xlsx file.

    :raises: SystemExit if some parameter names are not allowed
    """

    empty_cell_coords = list(zip(*np.where(pd.isnull(design_matrix))))

    empties = [
        "Realization {}, column {}".format(i, design_matrix.columns[j])
        for i, j in zip(*np.where(pd.isnull(design_matrix)))
    ]
    if len(empties) > 0:
        logger.warning("Design matrix contains empty cells {}".format(empties))

    # Look for initial or trailing whitespace in column headers. This
    # is disallowed as it can create user confusion and has no use-case.
    for col_header in design_matrix:
        if col_header != col_header.strip():
            raise SystemExit(
                'Column header "{}" contains initial or trailing whitespace.'.format(
                    col_header
                )
            )

    # pd.read_excel() will always allow duplicated column names, but will
    # change names of the columns so that if REAL is given twice, it will
    # occur as "REAL" and "REAL.1". Give warning if it looks like that
    # has happened.
    dot1columns = [
        col
        for col in design_matrix.columns
        if col.endswith(".1") and col.replace(".1", "") in design_matrix
    ]
    if dot1columns:
        logger.warning(
            "Column(s) {} are probably duplicated in design matrix".format(dot1columns)
        )

    denied_params = set(design_matrix.columns).intersection(set(DENYLIST))
    if denied_params:
        raise SystemExit(
            "Column name(s) {} is not allowed in design matrix.".format(denied_params)
        )

    return {cell_coord[0] for cell_coord in empty_cell_coords}


def _read_defaultssheet(xlsfilename, defaultssheetname):
    """
    Reads a XLSX file and tries to return a dataframe
    with defaults for design matrix parameters.

    A dataframe is always returned, possibly empty.
    Columns are always exactly named "keys" and "defaults"

    :raises: SystemExit if defaults sheet is non-empty but non-parsable
    """
    if defaultssheetname:
        default_df = _read_excel(xlsfilename, defaultssheetname, header=None)
        if default_df.empty:
            logger.info("Empty defaultssheet provided")
            default_df = pd.DataFrame(columns=[0, 1])
        if len(default_df.columns) < 2:
            raise SystemExit(
                "Defaults sheet must have exactly two columns, only one found"
            )
        if len(default_df.columns) > 2:
            warnings.warn(
                (
                    "DEPRECATION: You supplied more than two columns for "
                    "the default values. This is not supported and will stop "
                    "working in the future."
                ),
                UserWarning,
            )
            default_df = default_df[[0, 1]]  # Slicing columns
        # Look for initial or trailing whitespace in parameter names. This
        # is disallowed as it can create user confusion and has no use-case.
        for paramname in default_df.loc[:, 0]:
            if paramname != paramname.strip():
                raise SystemExit(
                    (
                        'Parameter name "{}" in default values contains '
                        "initial or trailing whitespace."
                    ).format(paramname)
                )

        denied_params = set(default_df.loc[:, 0]).intersection(set(DENYLIST_DEFAULTS))
        if denied_params:
            raise SystemExit(
                "Column name(s) {} is not allowed in design matrix defaults.".format(
                    denied_params
                )
            )

    else:
        logger.info("No defaultssheet provided, using empty dataframe")
        default_df = pd.DataFrame(columns=[0, 1])

    default_df.rename(columns={0: "keys", 1: "defaults"}, inplace=True)
    return default_df
