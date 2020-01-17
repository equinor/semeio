import logging
import sys

import numpy as np
import pandas as pd

import logging


# Filenames created/modified by this script. Don't change unless you understand the consequences.
_TARGET_FILE_TXT = "DESIGN2PARAMS.OK"
_DESIGN_MATRIX_TXT = "designmatrix.txt"
_DESIGN_PARAMETERS_TXT = "designparameters.txt"
_PARAMETERS_TXT = "parameters.txt"

logger = logging.getLogger(__name__)


def run(
    realization,
    xlsfilename,
    designsheetname,
    defaultssheetname,
    parametersfilename,
    log_level,
):
    """
    Reads out all file content from different files and create dataframes
    """
    logger.setLevel(log_level)

    design_matrix_sheet = _read_excel(xlsfilename, designsheetname)
    _validate_design_matrix(design_matrix_sheet)

    if defaultssheetname:
        default_sheet = _read_excel(xlsfilename, defaultssheetname, header=None)
    else:
        logger.info("No defaultssheet provided, using empty dataframe")
        default_sheet = pd.DataFrame(columns=[0, 1])

    default_sheet.rename(columns={0: "keys", 1: "defaults"}, inplace=True)

    try:
        parameters = pd.read_csv(parametersfilename, delimiter=" ", header=None)
    except IOError:
        logger.info(
            "No {} exists, creating a new, empty one.".format(parametersfilename)
        )
        parameters = pd.DataFrame(columns=[0, 1])
    parameters.rename(columns={0: "keys", 1: "parameters"}, inplace=True)

    _complete_parameters_file(
        realization, parameters, parametersfilename, design_matrix_sheet, default_sheet
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
    # get header / vals for choosen realization
    try:
        realization_values = pd.DataFrame(design_matrix_sheet.iloc[realization, 1:])
    except IndexError:
        raise SystemExit(
            "Provided realization arg {} does not exist in design matrix".format(
                realization
            )
        )
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

    # add new column with a combined parameters / data from the realization in design matrix
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
        return pd.read_excel(file_name, sheet_name, header=header)
    except IOError:
        raise SystemExit("File {} not found".format(file_name))
    except Exception as err:
        raise SystemExit(
            "File {} is probably not of correct type. Failed with exception '{}'".format(
                file_name, str(err)
            )
        )


def _validate_design_matrix(design_matrix):
    """
    Validate used design matrix
    :raises: SystemExit if design matrix contains empty headers
    :raises: SystemExit design matrix contains empty cells
    """
    # find column headers missing and raise exception if any
    unnamed = design_matrix.loc[:, design_matrix.columns.str.contains("^Unnamed")]
    column_indexes = [int(x.split(":")[1]) for x in unnamed.columns.values]
    if len(column_indexes) > 0:
        raise SystemExit(
            "Column headers not present in column {}".format(column_indexes)
        )

    # find empty cells and raise exception if any
    empties = [
        "Realization {}, key {}".format(i, j)
        for i, j in zip(*np.where(pd.isnull(design_matrix)))
    ]
    if len(empties) > 0:
        raise SystemExit("Design matrix contains empty cells {}".format(empties))
