"""Module for generating design matrices that can be run by DESIGN2PARAMS
and DESIGN_KW in FMU/ERT.


A DesignMatrix is a "God-object" that contains information about all info
used to generate design matrices, including one or several Sensitivities.


"""

from __future__ import annotations

import copy
from collections.abc import Hashable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import probabilit  # type: ignore[import-untyped]

import semeio
from semeio.fmudesign import design_distributions as design_dist
from semeio.fmudesign._excel_to_dict import _raise_if_duplicates
from semeio.fmudesign.quality_report import QualityReporter, print_corrmat
from semeio.fmudesign.utils import (
    find_max_realisations,
    map_dependencies,
    parameters_from_extern,
    printwarning,
    seeds_from_extern,
    to_numeric_safe,
)


class DesignMatrix:
    """Class for design matrix in FMU. Can contain a onebyone design
    or a full montecarlo design.

    Attributes:
        designvalues (pd.DataFrame): design matrix on standard fmu format
            contains columns 'REAL' (realization number), and if a onebyone
            design, also columns 'SENSNAME' and 'SENSCASE'
        defaultvalues (dict): default values for design
        backgroundvalues (pd.DataFrame): Used when background parameters are
            not constant. Either a set is sampled from specified distributions
            or they are read from a file.
    """

    def __init__(self, verbosity: int = 0, output_dir: Path | None = None) -> None:
        """
        Placeholders for:
        designvalues: dataframe with parameters that varies
        defaultvalues: dictionary of default/base case values
        backgroundvalues: dataframe with background parameters
        seedvalues: list of seed values
        verbosity: how much information to print
        output_dir: where to write debugging output and QC plots

        """
        self.designvalues: pd.DataFrame = pd.DataFrame(columns=["REAL"])
        self.defaultvalues: dict[Hashable, Any] = {}
        self.backgroundvalues: pd.DataFrame | None = None
        self.seedvalues: list[int] | None = None
        self.verbosity: int = verbosity
        self.output_dir: Path | None = output_dir

    def reset(self) -> None:
        """Resets DesignMatrix to empty. Necessary iin case method generate
        is used several times for same instance of DesignMatrix"""
        self.designvalues = pd.DataFrame(columns=["REAL"])
        self.defaultvalues = {}
        self.backgroundvalues = None
        self.seedvalues = None

    def generate(self, inputdict: Mapping[str, Any]) -> None:
        """Generating design matrix from input dictionary in specific
        format. Adding default values and background values if existing.
        Looping through sensitivities and adding them to designvalues.

        Args:
            inputdict (dict): input parameters for design
        """
        inputdict = copy.deepcopy(inputdict)

        if inputdict["designtype"] != "onebyone":
            raise ValueError(
                f"Generation of DesignMatrix only implemented for type 'onebyone', not {inputdict['designtype']}"
            )

        self.reset()  # Emptying if regenerating matrix

        rng = np.random.default_rng(seed=inputdict.get("distribution_seed"))

        self.defaultvalues = inputdict["defaultvalues"]

        max_reals = find_max_realisations(inputdict)

        # Reading or generating rms seed values
        if "seeds" in inputdict:
            self.add_seeds(inputdict["seeds"], max_reals)

        # If background values used - read or generate
        if "background" in inputdict:
            self.add_background(
                back_dict=inputdict["background"],
                max_values=max_reals,
                rng=rng,
                correlation_iterations=inputdict.get("correlation_iterations", 0),
            )

        sensitivity: Sensitivity

        self.designvalues["SENSNAME"] = None
        self.designvalues["SENSCASE"] = None
        current_real_index = 0
        for key in inputdict["sensitivities"]:
            sens = inputdict["sensitivities"][key]
            numreal = sens["numreal"] if "numreal" in sens else inputdict["repeats"]

            print(f"Generating sensitivity : {key}")

            if sens["senstype"] == "ref":
                sensitivity = SingleRealisationReference(key, verbosity=self.verbosity)
                sensitivity.generate(
                    realnums=range(current_real_index, current_real_index + numreal)
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))
                current_real_index += numreal
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "background":
                sensitivity = BackgroundSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(
                    realnums=range(current_real_index, current_real_index + numreal)
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))
                current_real_index += numreal
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "seed":
                if self.seedvalues is None:
                    raise ValueError(
                        "No seed values available to use for seed sensitivity"
                    )
                sensitivity = SeedSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(
                    realnums=range(current_real_index, current_real_index + numreal),
                    seedname=sens["seedname"],
                    seedvalues=self.seedvalues,
                    parameters=sens["parameters"],
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))
                current_real_index += numreal
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "scenario":
                sensitivity = ScenarioSensitivity(key, verbosity=self.verbosity)
                for casekey in sens["cases"]:
                    case = sens["cases"][casekey]
                    temp_case = ScenarioSensitivityCase(casekey)
                    temp_case.generate(
                        realnums=range(
                            current_real_index, current_real_index + numreal
                        ),
                        parameters=case,
                        seedvalues=self.seedvalues,
                    )
                    sensitivity.add_case(temp_case)
                    sensitivity.map_dependencies(sens.get("dependencies", {}))
                    current_real_index += numreal
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "dist":
                sensitivity = MonteCarloSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(
                    realnums=range(current_real_index, current_real_index + numreal),
                    parameters=sens["parameters"],
                    seedvalues=self.seedvalues,
                    corrdict=sens["correlations"],
                    rng=rng,
                    correlation_iterations=inputdict.get("correlation_iterations", 0),
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))
                current_real_index += numreal
                self._add_sensitivity(sensitivity)

            elif sens["senstype"] == "extern":
                sensitivity = ExternSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(
                    realnums=range(current_real_index, current_real_index + numreal),
                    filename=sens["extern_file"],
                    parameters=sens["parameters"],
                    seedvalues=self.seedvalues,
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))
                current_real_index += numreal
                self._add_sensitivity(sensitivity)
            print("Added sensitivity :", sensitivity.sensname)

            # MonteCarloSensitivity is special - it can produce debugging outputs
            is_montecarlo = isinstance(sensitivity, MonteCarloSensitivity)
            if is_montecarlo and self.verbosity > 0:
                sensitivity = cast(MonteCarloSensitivity, sensitivity)

                # Convert parameteters to a plottable string, e.g.
                # {"COSTS": "Normal(loc=0, scale=1)", ...}
                qr_vars = {
                    pname: f"{plist[0]}~("
                    + ", ".join(
                        f"dist_param{i}={v}" for (i, v) in enumerate(plist[1], 1)
                    )
                    + ")"
                    for (pname, plist) in sens["parameters"].items()
                }
                quality_reporter = QualityReporter(
                    df=sensitivity.sensvalues, variables=qr_vars
                )

                # Print to terminal
                quality_reporter.print_numeric()
                quality_reporter.print_discrete()
                for corr_name, df_corr in sensitivity.correlation_dfs_.items():
                    quality_reporter.print_correlation(corr_name, df_corr)

            if is_montecarlo and self.verbosity > 0 and self.output_dir is not None:
                sensitivity = cast(MonteCarloSensitivity, sensitivity)
                output_dir = self.output_dir / key
                quality_reporter.plot_columns(output_dir=output_dir)

                # Correlations
                for corr_name, df_corr in sensitivity.correlation_dfs_.items():
                    quality_reporter.plot_correlation(
                        corr_name, df_corr, output_dir=output_dir
                    )

        # Once all sensitivities have been added, complete the work
        if "background" in inputdict:
            self._fill_with_background_values()
        self._fill_with_defaultvalues()

        # Round columns in `self.designvalues` to desired precision
        self._set_decimals(inputdict)

        # Re-order columns
        start_cols = ["REAL", "SENSNAME", "SENSCASE", "RMS_SEED"]
        self.designvalues = self.designvalues[
            [col for col in start_cols if col in self.designvalues]
            + [col for col in self.designvalues if col not in start_cols]
        ]

        # Make all values numerical if possible
        self.designvalues = self.designvalues.map(to_numeric_safe)

    def to_xlsx(
        self,
        filename: str,
        designsheet: str = "DesignSheet01",
        defaultsheet: str = "DefaultValues",
    ) -> None:
        """Writing design matrix to excel workfbook on standard fmu format
        to be used in FMU/ERT by DESIGN2PARAMS and DESIGN_KW

        Args:
            filename (str): output filename (extension .xlsx)
            designsheet (str): name of excel sheet containing design matrix
                (optional, defaults to 'DesignSheet01')
            defaultsheet (str): name of excel sheet containing default
                values (optional, defaults to 'DefaultValues')
        """
        # Create folder for output file
        Path(filename).parent.mkdir(exist_ok=True, parents=True)

        if Path(filename).suffix != ".xlsx":
            filename = Path(filename).stem + ".xlsx"
            print(f"Warning: Missing .xlsx suffix. Changed to: {filename}")

        xlsxwriter = pd.ExcelWriter(filename, engine="openpyxl")
        self.designvalues.to_excel(
            xlsxwriter, sheet_name=designsheet, index=False, header=True
        )
        # Default values from OrderdDictionay to pandas dataframe
        defaults = pd.DataFrame(
            data=list(self.defaultvalues.items()),
            columns=["defaultparameters", "defaultvalue"],
        )
        defaults.to_excel(
            xlsxwriter, sheet_name=defaultsheet, index=False, header=False
        )

        version_info = pd.DataFrame(
            {
                "Description": ["Created using semeio version:", "Created on:"],
                "Value": [
                    semeio.__version__,
                    datetime.now().isoformat(sep=" ", timespec="seconds"),
                ],
            }
        )
        version_info.to_excel(xlsxwriter, sheet_name="Metadata", index=False)

        xlsxwriter.close()
        print(
            f"A total of {len(self.designvalues['REAL'])} realizations were generated"
        )
        print(f"Designmatrix written to {filename}")

    def add_seeds(self, seeds: str | None, max_reals: int) -> None:
        """Set RMS seed values.

        Configures seed values either by loading from an external file, generating
        default sequential seeds, or setting to None based on the input parameter.

        Args:
            seeds: Seed configuration. Can be:
                - "None" or None: Sets seedvalues to None
                - "default": Generates sequential seeds starting from 1000
                - str that represents a path to an existing file
            max_reals: Maximum number of seed values to generate or load
        """
        if seeds in {None, "None"}:
            self.seedvalues = None
            print("seeds is set to None in general_input")
        elif seeds and seeds.lower() == "default":
            self.seedvalues = [item + 1000 for item in range(max_reals)]
        elif seeds and Path(seeds).is_file():
            self.seedvalues = seeds_from_extern(seeds, max_reals)
        else:
            raise ValueError(
                "Valid choices for seeds are None, "
                '"default" or an existing filename. '
                "Neither was found in this case. seeds "
                f"had been specified as {seeds} ."
            )

    def add_background(
        self,
        back_dict: Mapping[str, Any] | None,
        max_values: int,
        rng: np.random.Generator,
        correlation_iterations: int = 0,
    ) -> None:
        """Adding background as specified in dictionary.
        Either from external file or from distributions in background
        dictionary

        Args:
            back_dict (dict): how to generate background values
            max_values (int): number of background values to generate
            rng (numpy.random.Generator): Random number generator instance
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
        """
        if back_dict is None:
            self.backgroundvalues = None
        elif "extern" in back_dict:
            print(f"Reading background values from: {back_dict['extern']}")
            self.backgroundvalues = parameters_from_extern(back_dict["extern"])
        elif "parameters" in back_dict:
            print("Generating background values from distributions.")
            self._add_dist_background(
                back_dict,
                max_values,
                rng,
                correlation_iterations=correlation_iterations,
            )

    def background_to_excel(
        self, filename: str, backgroundsheet: str = "Background"
    ) -> None:
        """Writing background values to an Excel spreadsheet

        Args:
            filename (str): output filename (extension .xlsx)
            backgroundsheet (str): name of excel sheet
        """
        if self.backgroundvalues is None:
            raise ValueError("No background values available to write to Excel")

        xlsxwriter = pd.ExcelWriter(filename, engine="openpyxl")
        self.backgroundvalues.to_excel(
            xlsxwriter, sheet_name=backgroundsheet, index=False, header=True
        )
        xlsxwriter.close()
        print(f"Backgroundvalues written to {filename}")

    def _add_sensitivity(
        self,
        sensitivity: Sensitivity,
    ) -> None:
        """Adding a sensitivity to the design

        Args:
            sensitivity of class Scenario, MonteCarlo or Extern
        """
        existing_values = self.designvalues
        new_values = sensitivity.sensvalues
        self.designvalues = pd.concat([existing_values, new_values])

    def _fill_with_background_values(self) -> None:
        """Substituting NaNs with background values if existing.
        background values not in design are added as separate columns
        """
        if self.backgroundvalues is not None:
            grouped = self.designvalues.groupby(["SENSNAME", "SENSCASE"], sort=False)
            result_values = pd.DataFrame()
            for sensname, case in grouped:
                temp_df = case.reset_index()
                temp_df.fillna(self.backgroundvalues, inplace=True)
                temp_df.set_index("index")
                for key in self.backgroundvalues:
                    if key not in case:
                        temp_df[key] = self.backgroundvalues[key]
                        if len(temp_df) > len(self.backgroundvalues):
                            raise ValueError(
                                "Provided number of background values "
                                f"{len(self.backgroundvalues)} is smaller than number"
                                f" of realisations for sensitivity {sensname}"
                            )
                    elif len(temp_df) > len(self.backgroundvalues):
                        print(
                            "Provided number of background values "
                            f"({len(self.backgroundvalues)}) is smaller than number"
                            f" of realisations for sensitivity {sensname}"
                            f" and parameter {key}. "
                            "Will be filled with default values."
                        )
                existing_values = result_values.copy()
                result_values = pd.concat([existing_values, temp_df])

            result_values = result_values.drop(["index"], axis=1)
            self.designvalues = result_values

    def _fill_with_defaultvalues(self) -> None:
        """Filling NaNs with default values"""
        for key in self.designvalues:
            if key in self.defaultvalues:
                self.designvalues[key] = self.designvalues[key].fillna(
                    self.defaultvalues[key]
                )
            elif key not in {"REAL", "SENSNAME", "SENSCASE", "RMS_SEED"}:
                raise LookupError(f"No defaultvalues given for parameter {key} ")

    def _add_dist_background(
        self,
        back_dict: Mapping[str, Any],
        numreal: int,
        rng: np.random.Generator,
        correlation_iterations: int,
    ) -> None:
        """Drawing background values from distributions
        specified in dictionary

        Args:
            back_dict (dict): parameters and distributions
            numreal (int): Number of samples to generate
            rng (numpy.random.Generator): Random number generator instance
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
        """

        mc_background = MonteCarloSensitivity("background")
        mc_background.generate(
            realnums=range(numreal),
            parameters=back_dict["parameters"],
            seedvalues=None,
            corrdict=back_dict["correlations"],
            rng=rng,
            correlation_iterations=correlation_iterations,
        )
        mc_backgroundvalues = mc_background.sensvalues.copy()

        # Reporting
        # Convert to plottable string: {"COSTS": "Normal(loc=0, scale=1)", ...}
        qr_vars = {
            pname: f"{plist[0]}~("
            + ", ".join(f"dist_param{i}={v}" for (i, v) in enumerate(plist[1], 1))
            + ")"
            for (pname, plist) in back_dict["parameters"].items()
        }
        quality_reporter = QualityReporter(df=mc_backgroundvalues, variables=qr_vars)

        # Print info to terminal
        if self.verbosity > 0:
            quality_reporter.print_numeric()
            quality_reporter.print_discrete()
            for corr_name, df_corr in mc_background.correlation_dfs_.items():
                quality_reporter.print_correlation(corr_name, df_corr)

        # Write plots to disk
        if self.verbosity > 0 and self.output_dir is not None:
            output_dir = self.output_dir / mc_background.sensname
            quality_reporter.plot_columns(output_dir=output_dir)

            # Correlations
            for corr_name, df_corr in mc_background.correlation_dfs_.items():
                quality_reporter.plot_correlation(
                    corr_name, df_corr, output_dir=output_dir
                )

        # Rounding of background values as specified
        if "decimals" in back_dict:
            for key in back_dict["decimals"]:
                if design_dist.is_number(mc_backgroundvalues[key].iloc[0]):
                    mc_backgroundvalues[key] = (
                        mc_backgroundvalues[key]
                        .astype(float)
                        .round(int(back_dict["decimals"][key]))
                    )
                else:
                    raise ValueError("Cannot round a string parameter")
        self.backgroundvalues = mc_backgroundvalues.copy()

    def _set_decimals(self, inputdict: Mapping[str, Any]) -> None:
        """Round to specified number of decimals.

        Args:
            inputdict (dictionary): input diction that might have a sub-dict
                                    with key "decimals". This sub-dict has
                                    (key, value)s are (param, decimals)
        """
        inputdict = copy.deepcopy(inputdict)

        # No decimal information => Nothing to do.
        if not inputdict.get("decimals", {}):
            return None

        # If there are dependencies (derived params) that are copies,
        # like TO := copy(FROM), then the new TO column must be rounded too.
        for sensdict in inputdict["sensitivities"].values():
            if not sensdict["dependencies"]:
                continue
            for from_param, from_dict in sensdict["dependencies"].items():
                for to_param in from_dict["to_params"]:
                    if not inputdict["decimals"].get(from_param, None):
                        continue
                    inputdict["decimals"][to_param] = inputdict["decimals"].get(
                        from_param, ""
                    )

        # Round each column
        dict_decimals = inputdict["decimals"]
        for key in self.designvalues:
            if key in dict_decimals:
                if design_dist.is_number(self.designvalues[key].iloc[0]):
                    self.designvalues[key] = (
                        self.designvalues[key]
                        .astype(float)
                        .round(int(dict_decimals[key]))
                    )
                else:
                    raise ValueError(f"Cannot round a string parameter {key}")


class Sensitivity:
    sensvalues: pd.DataFrame

    def __init__(self, sensname: str, verbosity: int = 0) -> None:
        """
        Args:
            sensname (str): Name of sensitivity. Defines SENSNAME in design matrix.
            verbosity (int): How much information to print. Non-negative integer.
        """
        self.sensname: str = sensname
        self.verbosity: int = verbosity

    def map_dependencies(self, dependencies: Mapping[str, Any]) -> Sensitivity:
        """Map the dependencies, mutating the dataframe `self.sensvalues`."""
        verbose = self.verbosity > 0  # Because the function takes a boolean
        self.sensvalues: pd.DataFrame = map_dependencies(
            self.sensvalues, dependencies=dependencies, verbose=verbose
        )
        return self


class SeedSensitivity(Sensitivity):
    """
    A seed sensitivity is normally the reference for one by one sensitivities,
    which all other sensitivities are compared to. All parameters will be at
    their default values. Only the RMS_SEED will be varying.

    It contains a list of seeds to be repeated for each sensitivity
    The parameter name is hardcoded to RMS_SEED
    It will be assigned the sensname 'p10_p90' which will be written to
    the SENSCASE column in the output.

    Attributes:
        sensname (str): name of sensitivity
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(
        self,
        realnums: range,
        seedname: str,
        seedvalues: Sequence[int],
        parameters: Mapping[str, Any] | None,
    ) -> None:
        """Generates parameter values for a seed sensitivity

        Args:
            realnums (list): list of integers with realization numbers
            seedname (str): name of seed parameter to add
            seedvalues (list): list of integer seedvalues
            parameters (dict): parameter names and
                distributions or values.
        """

        self.sensvalues = pd.DataFrame(index=realnums)
        self.sensvalues[seedname] = seedvalues[0 : len(realnums)]

        if parameters is not None:
            for key in parameters:
                dist_name = parameters[key][0].lower()
                constant = parameters[key][1]
                if dist_name != "const":
                    raise ValueError(
                        'A sensitivity of type "seed" can only have '
                        "additional parameters where dist_name is "
                        f'"const". Check sensitivity {self.sensname}"'
                    )
                self.sensvalues[key] = constant

        self.sensvalues["REAL"] = realnums
        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"


class SingleRealisationReference(Sensitivity):
    """
    The class is used in set-ups where one wants a single realisation
    containing only default values as a reference, but the realisation
    itself is not included in a sensitivity.
    Typically used when RMS_SEED is not a parameter.
    SENSCASE will be set to 'ref' in design matrix, to flag that it should be
    excluded as a sensitivity in the plot.

    Attributes:
        sensname (str): name of sensitivity
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(self, realnums: range) -> None:
        """Generates realisation number only

        Args:
            realnums (list): list of integers with realization numbers
        """
        self.sensvalues = pd.DataFrame(index=realnums)
        self.sensvalues["REAL"] = realnums
        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "ref"


class BackgroundSensitivity(Sensitivity):
    """
    The class is used in set-ups where one sensitivities
    are run on top of varying background parameters.
    Typically used when RMS_SEED is not a parameter, so the reference
    for tornadoplots will be the realisations with all parameters
    at their default values except the background parameters.
    SENSCASE will be set to 'p10_p90' in design matrix.

    Attributes:
        sensname (str): name of sensitivity
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(self, realnums: range) -> None:
        """Generates realisation number only

        Args:
            realnums (list): list of integers with realization numbers
        """
        self.sensvalues = pd.DataFrame(index=realnums)
        self.sensvalues["REAL"] = realnums
        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"


class ScenarioSensitivity(Sensitivity):
    """Each design can contain one or several single sensitivities of type
    Seed, MonteCarlo or Scenario.
    Each ScenarioSensitivity can contain 1-2 ScenarioSensitivityCases.

    The ScenarioSensitivity class is used for sensitivities where all
    realizatons in a ScenarioSensitivityCase have identical values
    but one or more parameter has a different values from the other
    ScenarioSensitivityCase.

    Exception is the seed value and the special case where
    varying background parameters are specified. Then these are varying
    within the case.

    Attributes:
        case1 (ScenarioSensitivityCase): first case, e.g. 'low case'
        case2 (ScenarioSensitivityCase): second case, e.g. 'high case'
        sensvalues (pd.DataFrame): design values for the sensitivity, containing
           1-2 cases
    """

    case1: ScenarioSensitivityCase | None = None
    case2: ScenarioSensitivityCase | None = None

    def add_case(self, senscase: ScenarioSensitivityCase) -> None:
        """
        Adds a ScenarioSensitivityCase instance
        to a ScenarioSensitivity object.

        Args:
            senscase (ScenarioSensitivityCase):
                Equals SENSCASE in design matrix.
        """
        if self.case1 is not None:  # Case 1 has been read, this is case2
            if (
                senscase.sensvalues is not None
                and "REAL" in senscase.sensvalues
                and "SENSCASE" in senscase.sensvalues
            ):
                self.case2 = senscase
                senscase.sensvalues["SENSNAME"] = self.sensname
                self.sensvalues = pd.concat(
                    [self.sensvalues, senscase.sensvalues], sort=True
                )
        elif (
            senscase.sensvalues is not None
            and "REAL" in senscase.sensvalues
            and "SENSCASE" in senscase.sensvalues
        ):
            self.case1 = senscase
            self.sensvalues = senscase.sensvalues.copy()
            self.sensvalues["SENSNAME"] = self.sensname


class ScenarioSensitivityCase(Sensitivity):
    """Each ScenarioSensitivity can contain one or
    two ScenarioSensitivityCases.

    The 1-2 cases are typically 'low' and 'high' cases for one or
    a set of  parameters, where all realisatons in
    the case have identical values except the seed value
    and in special cases specified background values which may
    vary within the case.

    One or two ScenarioSensitivityCase instances can be added to each
    ScenarioSensitivity object.

    Attributes:
        sensname (str): name of the sensitivity case,
            equals SENSCASE in design matrix.
        sensvalues (pd.DataFrame): parameters and values
            for the sensitivity with realisation numbers as index.

    """

    def generate(
        self,
        realnums: range,
        parameters: dict[str, Any],
        seedvalues: Sequence[int] | None,
    ) -> None:
        """Generate sensvalues for the ScenarioSensitivityCase

        Args:
            realnums (list): list of realizaton numbers for the case
            parameters (dict):
                dictionary with parameter names and values
            seeds (str): default or None
        """

        self.sensvalues = pd.DataFrame(columns=list(parameters.keys()), index=realnums)
        for key, value in parameters.items():
            self.sensvalues[key] = value
        self.sensvalues["REAL"] = realnums
        self.sensvalues["SENSCASE"] = self.sensname

        if seedvalues:
            self.sensvalues["RMS_SEED"] = seedvalues[: len(realnums)]


class MonteCarloSensitivity(Sensitivity):
    """
    For a MonteCarloSensitivity one or several parameters
    are drawn from specified distributions with or without correlations.
    A MonteCarloSensitivity can only contain
    one case, where the name SENSCASE is automatically set to 'p10_p90' in the
    design matrix to flag that p10_p90 should be calculated in TornadoPlot.

    Attributes:
        sensname (str):  name for the sensitivity.
            Equals SENSNAME in design matrix.
        sensvalues (pd.DataFrame):  parameters and values for the sensitivity
            with realisation numbers as index.
    """

    def generate(
        self,
        *,
        realnums: range,
        parameters: dict[str, Any],
        seedvalues: Sequence[int] | None,
        corrdict: Mapping[str, Any] | None,
        rng: np.random.Generator,
        correlation_iterations: int = 0,
    ) -> None:
        """Generates parameter values by drawing from defined distributions.

        Args:
            realnums (range): range object containing realization numbers
            parameters (dict): dictionary of parameters and distributions
            seeds (str): default or None
            corrdict (dict): Configuration for correlated parameters. Contains:
                - 'inputfile': Path to Excel file with correlation matrices
                - 'sheetnames': List of sheet names, where each sheet contains a correlation matrix
                If None, parameters are treated as uncorrelated.
            rng (numpy.random.Generator): Random number generator instance
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
        """
        self.sensvalues = pd.DataFrame(columns=list(parameters.keys()), index=realnums)
        self.correlation_dfs_ = {}  # Store correlation matrices (dataframes)
        numreals = len(realnums)

        if numreals < 0:
            raise ValueError(f"Got < 0 samples ({numreals=})")

        distr_by_name = {}
        for param_name, (dist_name, dist_params, _) in parameters.items():
            # Convert to a probabilit Distribution object
            distr = design_dist.to_probabilit(
                distname=dist_name, dist_parameters=dist_params
            )
            distr_by_name[param_name] = distr

        # Create a dummy NoOp node for sampling each parent distribution
        expression = probabilit.modeling.NoOp(*distr_by_name.values())

        if corrdict:
            # Create an iterator over correlation groups from the main sheet
            df_params = (
                pd.DataFrame.from_dict(
                    parameters,
                    orient="index",
                    columns=["dist_name", "dist_params", "corr_sheet"],
                )
                .reset_index()
                .rename(columns={"index": "param_name"})
                .assign(corr_sheet=lambda df: df.corr_sheet.fillna("nocorr"))
            )

            corr_groups = dict(iter(df_params.groupby("corr_sheet")))
            corr_groups.pop("nocorr", None)

            for corr_group_name, corr_group in corr_groups.items():
                corr_group_name = cast(str, corr_group_name)

                # Skip nocorr
                if corr_group_name == "nocorr":
                    continue

                # A single correlation - print warning and skip it
                if len(corr_group) == 1:
                    printwarning(corr_group_name)
                    continue

                # Read correlation matrix and convert it to a proper matrix
                df_correlations = design_dist.read_correlations(
                    excel_filename=corrdict["inputfile"], corr_sheet=corr_group_name
                )
                multivariate_parameters = list(df_correlations.index.values)
                correlations = df_correlations.values

                print(
                    f"Sampling parameters in {corr_group_name!r}: {multivariate_parameters}"
                )

                # Get the nearest correlation matrix
                nearest = probabilit.correlation.nearest_correlation_matrix(
                    correlations, weights=None, eps=1e-6, verbose=False
                )
                if not np.allclose(correlations, nearest):
                    print(
                        f"\nWarning: Correlation matrix {corr_group_name!r} is inconsistent"
                    )
                    print("Requirements:")
                    print("  - All diagonal elements must be 1")
                    print("  - All elements must be between -1 and 1")
                    print("  - The matrix must be positive semi-definite")
                    print("\nInput correlation matrix:")
                    print_corrmat(df_correlations)
                    df_correlations.loc[:] = nearest
                    print("\nAdjusted to nearest consistent correlation matrix:")
                    print_corrmat(df_correlations)

                corrvars = [distr_by_name[name] for name in df_correlations.columns]
                expression.correlate(*corrvars, corr_mat=df_correlations.values)
                self.correlation_dfs_[corr_group_name] = df_correlations

        # Either do ImanConover followed by Permutation, or simply ImanConover
        if correlation_iterations > 0:
            # TODO: It is possible to let the user set the correlation type
            # if this is of interest. But for now we assume that users care about
            # pearson correlation, not spearman (rank) correlation.
            correlator = probabilit.correlation.Composite(
                iterations=correlation_iterations,
                correlation_type="pearson",
                random_state=rng,
                verbose=False,
            )
        else:
            correlator = probabilit.correlation.ImanConover()

        # Sample the dummy node - this samples every parent and populates "samples_"
        expression.sample(
            size=numreals, random_state=rng, method="lhs", correlator=correlator
        )

        for distr_name, distr_obj in distr_by_name.items():
            samples = distr_obj.samples_
            is_numeric = issubclass(samples.dtype.type, np.number)
            if is_numeric and not np.all(np.isfinite(distr_obj.samples_)):
                raise ValueError(
                    f"Sampling produced non-finite values in {distr_name}={distr_obj}\n"
                    "Please review the parameters in the distribution."
                )

            # Discrete distributions are handled in a special way. We map them
            # to Uniform distributions, sample in [0, 1), then map those samples
            # back to the categorical values AFTER sampling. This is so that we
            # can "induce correlations" between categorical values.
            if hasattr(distr_obj, "_values"):
                probabilities = getattr(distr_obj, "_probabilities", None)
                samples = design_dist.quantiles_to_values(
                    quantiles=samples,
                    values=distr_obj._values,
                    probabilities=probabilities,
                )

            self.sensvalues = self.sensvalues.assign(**{distr_name: samples})
            print(f"Wrote {numreals} samples from {distr_name!r}")

        if self.sensname != "background":
            self.sensvalues["REAL"] = realnums
            self.sensvalues["SENSNAME"] = self.sensname
            self.sensvalues["SENSCASE"] = "p10_p90"
            if "RMS_SEED" not in self.sensvalues and seedvalues:
                self.sensvalues["RMS_SEED"] = seedvalues[: len(realnums)]

        null_columns = self.sensvalues.isnull().any(axis=0)
        if null_columns.any():
            cols_w_null = list(null_columns.loc[lambda ser: ser].index)
            raise ValueError(f"Found NaN values in columns: {cols_w_null}")


class ExternSensitivity(Sensitivity):
    """
    Used when reading parameter values from a file
    Assumed to be used with monte carlo type sensitivities and
    will hence write 'p10_p90' as SENSCASE in output designmatrix

    Attributes:
        sensname (str): Name of sensitivity.
            Defines SENSNAME in design matrix
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(
        self,
        realnums: range,
        filename: str,
        parameters: list[str],
        seedvalues: Sequence[int] | None,
    ) -> None:
        """Reads parameter values for a monte carlo sensitivity
        from file

        Args:
            realnums (list): list of integers with realization numbers
            filename (str): path where to read values from
            parameters (list): list with parameter names
            seeds (str): default or None
        """
        _raise_if_duplicates(parameters)
        self.sensvalues = pd.DataFrame(columns=parameters, index=realnums)
        extern_values = parameters_from_extern(filename)
        if len(realnums) > len(extern_values):
            raise ValueError(
                f"Number of realisations {len(realnums)} specified for "
                f"sensitivity {self.sensname} is larger than rows in "
                f"file {filename}"
            )
        for param in parameters:
            if param in extern_values:
                self.sensvalues[param] = list(extern_values[param][: len(realnums)])
            else:
                raise ValueError(f"Parameter {param} not in external file")
        self.sensvalues["REAL"] = realnums
        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"

        if seedvalues:
            self.sensvalues["RMS_SEED"] = seedvalues[: len(realnums)]
