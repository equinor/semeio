"""Module for reading excel file with input for generation of
a design matrix and converting to an OrderedDict that can be
read by semeio.fmudesign.DesignMatrix.generate
"""

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np
import openpyxl
import pandas as pd
import yaml


def excel2dict_design(
    input_filename: str, sheetnames: Mapping[str, Any] | None = None
) -> OrderedDict[str, Any]:
    """Read excel file with input to design setup
    Currently only specification of
    onebyone design is implemented

    Args:
        input_filename (str): Name of excel input file
        sheetnames (dict): Dictionary of worksheet names to load
            information from. Supported keys: general_input, defaultvalues,
            and designinput.

    Returns:
        OrderedDict on format for DesignMatrix.generate
    """
    if not sheetnames or "general_input" not in sheetnames:
        gen_input_sheet = _find_geninput_sheetname(input_filename)
    else:
        gen_input_sheet = sheetnames["general_input"]

    generalinput = pd.read_excel(
        input_filename, gen_input_sheet, header=None, index_col=0, engine="openpyxl"
    )
    generalinput.dropna(axis=0, how="all", inplace=True)
    generalinput.dropna(axis=1, how="all", inplace=True)

    if str(generalinput[1]["designtype"]) == "onebyone":
        returndict = _excel2dict_onebyone(input_filename, sheetnames)
    elif "seeds" in generalinput[1]:
        raise ValueError(
            "The 'seeds' parameter has been deprecated and is no longer supported. "
            "Use 'rms_seeds' instead"
        )
    else:
        raise ValueError(
            "Generation of DesignMatrix only "
            "implemented for type onebyone "
            "In general_input designtype was "
            "set to {}".format(str(generalinput[1]["designtype"]))
        )
    return returndict


def inputdict_to_yaml(inputdict: Mapping[str, Any], filename: str) -> None:
    """Write inputdict to yaml format

    Args:
        inputdict (OrderedDict)
        filename (str): path for where to write file
    """
    with open(filename, "w", encoding="utf-8") as stream:
        yaml.dump(inputdict, stream)


def _find_geninput_sheetname(input_filename: str) -> str:
    """Finding general input sheet, allowing for name
    variations."""
    xlsx = openpyxl.load_workbook(input_filename, read_only=True, keep_links=False)
    sheets = xlsx.sheetnames
    general_input_sheet: list[str] = []
    for sheet in sheets:
        if sheet in [
            "general_input",
            "generalinput",
            "GeneralInput",
            "Generalinput",
            "General_Input",
            "General_input",
        ]:
            general_input_sheet.append(sheet)

    if len(general_input_sheet) > 1:
        raise ValueError(
            f"More than one sheet with general input. Sheetnames are {general_input_sheet} "
        )

    if not general_input_sheet:
        raise ValueError(
            f"No general_input sheet provided in Excel file {input_filename} "
        )

    return general_input_sheet[0]


def _find_onebyone_defaults_sheet(input_filename: str) -> str:
    """Finds correct sheet name for default values to use when parsing
    excel file.

    Returns:
        string, name of a sheet in the excel file
    """
    xlsx = openpyxl.load_workbook(input_filename, read_only=True, keep_links=False)
    sheets = xlsx.sheetnames

    default_values_sheet: list[str] = []

    for sheet in sheets:
        if sheet in [
            "default_values",
            "defaultvalues",
            "DefaultValues",
            "Defaultvalues",
            "Default_Values",
            "Default_values",
        ]:
            default_values_sheet.append(sheet)
    if len(default_values_sheet) > 1:
        raise ValueError(
            f"More than one sheet with default values. Sheetnames are {default_values_sheet} "
        )

    if not default_values_sheet:
        raise ValueError(
            f"No defaultvalues sheet provided in Excel file {input_filename} "
        )

    return default_values_sheet[0]


def _find_onebyone_input_sheet(input_filename: str) -> str:
    """Finds correct sheet name for input to use when parsing excel file.

    Returns:
        string, name of a sheet in the excel file
    """
    xlsx = openpyxl.load_workbook(input_filename, read_only=True, keep_links=False)
    sheets = xlsx.sheetnames

    design_input_sheet: list[str] = []

    for sheet in sheets:
        if sheet in [
            "design_input",
            "designinput",
            "DesignInput",
            "Designinput",
            "Design_Input",
            "Design_input",
        ]:
            design_input_sheet.append(sheet)
    if len(design_input_sheet) > 1:
        raise ValueError(
            f"More than one sheet with design inputSheetnames are {design_input_sheet} "
        )

    if not design_input_sheet:
        raise ValueError(
            f"No designinput sheet provided in Excel file {input_filename} "
        )
    return design_input_sheet[0]


def _check_designinput(dsgn_input: pd.DataFrame) -> None:
    """Checks for valid input in designinput sheet"""

    # Check for duplicate sensnames
    sensitivity_names = []
    for row in dsgn_input.itertuples():
        if _has_value(row.sensname):
            if row.sensname in sensitivity_names:
                raise ValueError(
                    f"sensname '{row.sensname}' was found on more than one row in designinput "
                    "sheet. Two sensitivities can not share the same sensname. "
                    "Please correct this and rerun"
                )
            sensitivity_names.append(row.sensname)


def _check_for_mixed_sensitivities(sens_name: str, sens_group: pd.DataFrame) -> None:
    """Checks for valid input in designinput sheet. A sensitivity cannot contain
    two different sensitivity types"""

    types = sens_group.groupby("type", sort=False)
    if len(types) > 1:
        raise ValueError(
            f"The sensitivity with sensname '{sens_name}' in designinput sheet contains more "
            "than one sensitivity type. For each sensname all parameters must be "
            "specified using the same type (seed, scenario, dist, ref, background, "
            "extern)"
        )


def _excel2dict_onebyone(
    input_filename: str, sheetnames: Mapping[str, Any] | None = None
) -> OrderedDict[str, Any]:
    """Reads specification for onebyone design

    Args:
        input_filename(path): path to excel workbook
        sheetnames (dict): Dictionary of worksheet names to load
            information from. Supported keys: general_input, defaultvalues,
            and designinput.

    Returns:
        OrderedDict on format for DesignMatrix.generate
    """

    seedname = "RMS_SEED"
    inputdict: OrderedDict[str, Any] = OrderedDict()

    if not sheetnames or "general_input" not in sheetnames:
        gen_input_sheet = _find_geninput_sheetname(input_filename)
    else:
        gen_input_sheet = sheetnames["general_input"]

    if not sheetnames or "designinput" not in sheetnames:
        design_inp_sheet = _find_onebyone_input_sheet(input_filename)
    else:
        design_inp_sheet = sheetnames["designinput"]

    if not sheetnames or "defaultvalues" not in sheetnames:
        default_val_sheet = _find_onebyone_defaults_sheet(input_filename)
    else:
        default_val_sheet = sheetnames["defaultvalues"]

    # Read general input
    generalinput = pd.read_excel(
        input_filename, gen_input_sheet, header=None, index_col=0, engine="openpyxl"
    )
    generalinput.dropna(axis=0, how="all", inplace=True)
    generalinput.dropna(axis=1, how="all", inplace=True)

    inputdict["designtype"] = generalinput[1]["designtype"]

    if "rms_seeds" in generalinput[1]:
        if str(generalinput[1]["rms_seeds"]) == "None":
            inputdict["seeds"] = None
        else:
            inputdict["seeds"] = generalinput[1]["rms_seeds"]
    else:
        inputdict["seeds"] = None

    if "repeats" not in generalinput[1]:
        raise LookupError('"repeats" must be specified in general_input sheet')

    inputdict["repeats"] = generalinput[1]["repeats"]

    if "distribution_seed" in generalinput[1]:
        if str(generalinput[1]["distribution_seed"]) == "None":
            inputdict["distribution_seed"] = None
        else:
            inputdict["distribution_seed"] = generalinput[1]["distribution_seed"]
    else:
        inputdict["distribution_seed"] = None

    # Read background
    if "background" in generalinput.index:
        inputdict["background"] = OrderedDict()
        if generalinput[1]["background"].endswith("csv") or generalinput[1][
            "background"
        ].endswith("xlsx"):
            inputdict["background"]["extern"] = generalinput[1]["background"]
        elif str(generalinput[1]["background"]) == "None":
            inputdict["background"] = None
        else:
            inputdict["background"] = _read_background(
                input_filename, generalinput[1]["background"]
            )
    else:
        inputdict["background"] = None

    # Read default values
    inputdict["defaultvalues"] = _read_defaultvalues(input_filename, default_val_sheet)

    # Read input for sensitivities
    inputdict["sensitivities"] = OrderedDict()
    designinput = pd.read_excel(input_filename, design_inp_sheet, engine="openpyxl")
    designinput.dropna(axis=0, how="all", inplace=True)
    designinput = designinput.loc[
        :, ~designinput.columns.astype(str).str.contains("^Unnamed")
    ]

    # First column with parameter names should have spaces stripped,
    # but we need to preserve NaNs:
    not_nan_sensnames = ~designinput["sensname"].isnull()
    designinput.loc[not_nan_sensnames, "sensname"] = (
        designinput.loc[not_nan_sensnames, "sensname"].astype(str).str.strip()
    )

    _check_designinput(designinput)

    designinput["sensname"] = designinput["sensname"].ffill()

    # Read dependencies
    if "dependencies" in designinput:
        inputdict["dependencies"] = OrderedDict()
        for row in designinput.itertuples():
            if _has_value(row.dependencies):
                inputdict["dependencies"][row.param_name] = _read_dependencies(
                    input_filename,
                    row.dependencies,  # type: ignore[arg-type]
                    row.param_name,  # type: ignore[arg-type]
                )

    # Read decimals
    if "decimals" in designinput:
        inputdict["decimals"] = OrderedDict()
        for row in designinput.itertuples():
            if _has_value(row.decimals) and _is_int(row.decimals):  # type: ignore[arg-type]
                inputdict["decimals"][row.param_name] = int(row.decimals)  # type: ignore[arg-type]

    grouped = designinput.groupby("sensname", sort=False)

    # Read each sensitivity
    for sensname, group in grouped:
        _check_for_mixed_sensitivities(
            sensname,  # type: ignore[arg-type]
            group,
        )

        sensdict: OrderedDict[str, Any] = OrderedDict()

        if group["type"].iloc[0] == "ref":
            sensdict["senstype"] = "ref"

        elif group["type"].iloc[0] == "background":
            sensdict["senstype"] = "background"

        elif group["type"].iloc[0] == "seed":
            sensdict["seedname"] = seedname
            sensdict["senstype"] = "seed"
            if _has_value(group["param_name"].iloc[0]):
                sensdict["parameters"] = _read_constants(group)
            else:
                sensdict["parameters"] = None

        elif group["type"].iloc[0] == "scenario":
            sensdict = _read_scenario_sensitivity(group)
            sensdict["senstype"] = "scenario"

        elif group["type"].iloc[0] == "dist":
            sensdict["senstype"] = "dist"
            sensdict["parameters"] = _read_dist_sensitivity(group)
            sensdict["correlations"] = None
            if "corr_sheet" in group:
                sensdict["correlations"] = _read_correlations(group, input_filename)

        elif group["type"].iloc[0] == "extern":
            sensdict["extern_file"] = str(group["extern_file"].iloc[0])
            sensdict["senstype"] = "extern"
            sensdict["parameters"] = list(group["param_name"])

        else:
            raise ValueError(
                f"Sensitivity {sensname} does not have a valid sensitivity type"
            )

        if "numreal" in group and _has_value(group["numreal"].iloc[0]):
            # Using default number of realisations:
            # 'repeats' from general_input sheet
            sensdict["numreal"] = int(group["numreal"].iloc[0])

        inputdict["sensitivities"][str(sensname)] = sensdict

    return inputdict


def _read_defaultvalues(filename: str, sheetname: str) -> OrderedDict[str, Any]:
    """Reads defaultvalues, also used as values for
    reference/base case

    Args:
        filename(path): path to excel file
        sheetname (string): name of defaultsheet

    Returns:
        OrderedDict with defaultvalues (parameter, value)
    """
    default_dict: OrderedDict[str, Any] = OrderedDict()
    default_df = pd.read_excel(
        filename, sheetname, header=0, index_col=0, engine="openpyxl"
    )
    default_df.dropna(axis=0, how="all", inplace=True)
    default_df = default_df.loc[
        :, ~default_df.columns.astype(str).str.contains("^Unnamed")
    ]

    # Strip spaces before and after parameter names, if they are there
    # it is probably invisible user errors in Excel.

    default_df.index = pd.Index(
        [
            paramname.strip() if isinstance(paramname, str) else paramname
            for paramname in default_df.index
        ]
    )
    for row in default_df.itertuples():
        if str(row[0]) in default_dict:
            print(
                f"WARNING: The default value '{row[0]}' "
                f"is listed twice in the sheet '{sheetname}'. "
                "Only the first entry will be used in output file"
            )
        else:
            default_dict[str(row[0])] = row[1]
    return default_dict


def _read_dependencies(
    filename: str, sheetname: str, from_parameter: str
) -> OrderedDict[str, Any]:
    """Reads parameters that are set from other parameters

    Args:
        filename(path): path to excel file
        sheetname (string): name of dependency sheet

    Returns:
        OrderedDict with design parameter, dependent parameters
        and values
    """
    depend_dict: OrderedDict[str, Any] = OrderedDict()
    depend_df = pd.read_excel(
        filename, sheetname, dtype=str, na_values="", engine="openpyxl"
    )
    depend_df.dropna(axis=0, how="all", inplace=True)
    depend_df = depend_df.loc[
        :, ~depend_df.columns.astype(str).str.contains("^Unnamed")
    ]

    if from_parameter in depend_df:
        depend_dict["from_values"] = depend_df[from_parameter].tolist()
        depend_dict["to_params"] = OrderedDict()
        for key in depend_df:
            if key != from_parameter:
                depend_dict["to_params"][key] = depend_df[key].tolist()
    else:
        raise ValueError(
            f"Parameter {from_parameter} specified to have derived parameters, "
            f"but the sheet specifying the dependencies {sheetname} does "
            "not contain the input parameter. "
        )
    return depend_dict


def _read_background(inp_filename: str, bck_sheet: str) -> OrderedDict[str, Any]:
    """Reads excel sheet with background parameters and distributions

    Args:
        inp_filename (path): path to Excel workbook
        bck_sheet (str): name of sheet with background parameters

    Returns:
        OrderedDict with parameter names and distributions
    """
    backdict: OrderedDict[str, Any] = OrderedDict()
    paramdict: OrderedDict[str, Any] = OrderedDict()
    bck_input = pd.read_excel(inp_filename, bck_sheet, engine="openpyxl")
    bck_input.dropna(axis=0, how="all", inplace=True)
    bck_input = bck_input.loc[
        :, ~bck_input.columns.astype(str).str.contains("^Unnamed")
    ]

    backdict["correlations"] = None
    if "corr_sheet" in bck_input:
        backdict["correlations"] = _read_correlations(bck_input, inp_filename)

    if "dist_param1" not in bck_input.columns.values:
        bck_input["dist_param1"] = float("NaN")
    if "dist_param2" not in bck_input.columns.values:
        bck_input["dist_param2"] = float("NaN")
    if "dist_param3" not in bck_input.columns.values:
        bck_input["dist_param3"] = float("NaN")
    if "dist_param4" not in bck_input.columns.values:
        bck_input["dist_param4"] = float("NaN")

    for row in bck_input.itertuples():
        if not _has_value(row.param_name):
            raise ValueError(
                "Background parameters specified "
                "where one line has empty parameter "
                "name "
            )
        if not _has_value(row.dist_param1):
            raise ValueError(
                f"Parameter {row.param_name} has been input "
                "in background sheet but with empty "
                "first distribution parameter "
            )
        if not _has_value(row.dist_param2) and _has_value(row.dist_param3):
            raise ValueError(
                f"Parameter {row.param_name} has been input in "
                "background sheet with "
                'value for "dist_param3" while '
                '"dist_param2" is empty. This is not '
                "allowed"
            )
        if not _has_value(row.dist_param3) and _has_value(row.dist_param4):
            raise ValueError(
                f"Parameter {row.param_name} has been input in "
                "background sheet with "
                'value for "dist_param4" while '
                '"dist_param3" is empty. This is not '
                "allowed"
            )
        distparams = [
            item
            for item in [
                row.dist_param1,
                row.dist_param2,
                row.dist_param3,
                row.dist_param4,
            ]
            if _has_value(item)
        ]
        if "corr_sheet" in bck_input:
            corrsheet = None if not _has_value(row.corr_sheet) else row.corr_sheet
        else:
            corrsheet = None
        paramdict[str(row.param_name)] = [str(row.dist_name), distparams, corrsheet]
    backdict["parameters"] = paramdict

    if "decimals" in bck_input:
        decimals: OrderedDict[str, Any] = OrderedDict()
        for row in bck_input.itertuples():
            if _has_value(row.decimals) and _is_int(row.decimals):  # type: ignore[arg-type]
                decimals[row.param_name] = int(row.decimals)  # type: ignore[arg-type, index]
        backdict["decimals"] = decimals

    return backdict


def _read_scenario_sensitivity(sensgroup: pd.DataFrame) -> OrderedDict[str, Any]:
    """Reads parameters and values
    for scenario sensitivities
    """
    sdict: OrderedDict[str, Any] = OrderedDict()
    sdict["cases"] = OrderedDict()
    casedict1: OrderedDict[str, Any] = OrderedDict()
    casedict2: OrderedDict[str, Any] = OrderedDict()

    if not _has_value(sensgroup["senscase1"].iloc[0]):
        raise ValueError(
            "Sensitivity {} has been input "
            "as a scenario sensitivity, but "
            "without a name in senscase1 column.".format(sensgroup["sensname"].iloc[0])
        )

    for row in sensgroup.itertuples():
        if not _has_value(row.param_name):
            raise ValueError(
                f"Scenario sensitivity {row.sensname} specified "
                "where one line has empty parameter "
                "name "
            )
        if not _has_value(row.value1):
            raise ValueError(
                f"Parameter {row.param_name} har been input "
                'as type "scenario" but with empty '
                "value in value1 column "
            )
        casedict1[str(row.param_name)] = row.value1

    if _has_value(sensgroup["senscase2"].iloc[0]):
        for row in sensgroup.itertuples():
            if not _has_value(row.value2):
                raise ValueError(
                    "Sensitivity {} has been input "
                    "with a name in senscase2 column "
                    "but without a value for parameter {} "
                    "in value2 column.".format(
                        sensgroup["sensname"].iloc[0], row.param_name
                    )
                )
            casedict2[str(row.param_name)] = row.value2
        sdict["cases"][str(sensgroup["senscase1"].iloc[0])] = casedict1
        sdict["cases"][str(sensgroup["senscase2"].iloc[0])] = casedict2
    else:
        for row in sensgroup.itertuples():
            if _has_value(row.value2):
                raise ValueError(
                    "Sensitivity {} has been input "
                    "with a value for parameter {} "
                    "in value2 column "
                    "but without a name for the scenario "
                    "in senscase2 column.".format(
                        sensgroup["sensname"].iloc[0], row.param_name
                    )
                )
        sdict["cases"][str(sensgroup["senscase1"].iloc[0])] = casedict1
    return sdict


def _read_constants(sensgroup: pd.DataFrame) -> OrderedDict[str, Any]:
    """Reads constants to be used together with
    seed sensitivity"""
    if "dist_param1" not in sensgroup.columns.values:
        sensgroup["dist_param1"] = float("NaN")
    paramdict: OrderedDict[str, Any] = OrderedDict()
    for row in sensgroup.itertuples():
        if not _has_value(row.dist_param1):
            raise ValueError(
                f"Parameter name {row.param_name} has been input "
                'in a sensitivity of type "seed". \n'
                f"If {row.param_name} was meant to be the name of "
                "the seed parameter, this is "
                "unfortunately not allowed. "
                "The seed parameter name is standardised "
                "to RMS_SEED and should not be specified.\n "
                "If you instead meant to specify a constant "
                "value for another parameter in the seed "
                'sensitivity, please remember "const" in '
                'dist_name and a value in "dist_param1". '
            )
        distparams = row.dist_param1
        paramdict[str(row.param_name)] = [str(row.dist_name), distparams]
    return paramdict


def _read_dist_sensitivity(sensgroup: pd.DataFrame) -> OrderedDict[str, Any]:
    """Reads parameters and distributions
    for monte carlo sensitivities
    """
    if "dist_param1" not in sensgroup.columns.values:
        sensgroup["dist_param1"] = float("NaN")
    if "dist_param2" not in sensgroup.columns.values:
        sensgroup["dist_param2"] = float("NaN")
    if "dist_param3" not in sensgroup.columns.values:
        sensgroup["dist_param3"] = float("NaN")
    if "dist_param4" not in sensgroup.columns.values:
        sensgroup["dist_param4"] = float("NaN")
    paramdict: OrderedDict[str, Any] = OrderedDict()
    for row in sensgroup.itertuples():
        if not _has_value(row.param_name):
            raise ValueError(
                f"Dist sensitivity {row.sensname} specified "
                "where one line has empty parameter "
                "name "
            )
        if not _has_value(row.dist_param1):
            raise ValueError(
                f"Parameter {row.param_name} has been input "
                'as type "dist" but with empty '
                "first distribution parameter "
            )
        if not _has_value(row.dist_param2) and _has_value(row.dist_param3):
            raise ValueError(
                f"Parameter {row.param_name} has been input with "
                'value for "dist_param3" while '
                '"dist_param2" is empty. This is not '
                "allowed"
            )
        if not _has_value(row.dist_param3) and _has_value(row.dist_param4):
            raise ValueError(
                f"Parameter {row.param_name} has been input with "
                'value for "dist_param4" while '
                '"dist_param3" is empty. This is not '
                "allowed"
            )
        distparams = [
            item
            for item in [
                row.dist_param1,
                row.dist_param2,
                row.dist_param3,
                row.dist_param4,
            ]
            if _has_value(item)
        ]
        if "corr_sheet" in sensgroup:
            corrsheet = None if not _has_value(row.corr_sheet) else row.corr_sheet
        else:
            corrsheet = None
        paramdict[str(row.param_name)] = [str(row.dist_name), distparams, corrsheet]

    return paramdict


def _read_correlations(
    sensgroup: pd.DataFrame, inputfile: str
) -> OrderedDict[str, Any] | None:
    if "corr_sheet" in sensgroup:
        if not sensgroup["corr_sheet"].dropna().empty:
            correlations: OrderedDict[str, Any] = OrderedDict()
            correlations["inputfile"] = inputfile
            correlations["sheetnames"] = []
            for _index, row in sensgroup.iterrows():
                if (
                    _has_value(row["corr_sheet"])
                    and row["corr_sheet"] not in correlations["sheetnames"]
                ):
                    correlations["sheetnames"].append(row["corr_sheet"])
        else:
            return None
    else:
        return None

    return correlations


def _has_value(value: Any) -> bool:  # noqa: ANN401
    """Returns False only if the argument is np.nan"""
    try:
        return not np.isnan(value)
    except TypeError:
        return True


def _is_int(teststring: str) -> bool:
    """Test if a string can be parsed as a float"""
    try:
        if not np.isnan(int(teststring)):
            return (float(teststring) % 1) == 0
        return False  # It was a "number", but it was NaN.
    except ValueError:
        return False
