"""Module for generating design matrices that can be run by DESIGN2PARAMS
and DESIGN_KW in FMU/ERT.
"""

import contextlib
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

# CVXPY prints error messages about incompatible ortools version during import.
# Since we use the SCS solver and not GLOP/PDLP (which need ortools), these errors
# are irrelevant and would only confuse users. We suppress them by redirecting
# stdout/stderr during import.
# https://github.com/cvxpy/cvxpy/issues/2470
with (
    open(os.devnull, "w") as devnull,
    contextlib.redirect_stdout(devnull),
    contextlib.redirect_stderr(devnull),
):
    import cvxpy as cp

import numpy as np
import pandas as pd
import scipy
from scipy.stats import qmc

import semeio.fmudesign
from semeio.fmudesign import design_distributions as design_dist
from semeio.fmudesign.iman_conover import ImanConover


def is_consistent_correlation_matrix(matrix):
    """
    Check if a matrix is a consistent correlation matrix.

    A correlation matrix is consistent if it has:
    1. All diagonal elements equal to 1
    2. Is positive semidefinite (all eigenvalues ≥ 0)

    Args:
        matrix: numpy array representing the correlation matrix

    Returns:
        bool: True if matrix is a consistent correlation matrix, False otherwise
    """
    # Check if diagonal elements are 1
    if not np.allclose(np.diagonal(matrix), 1):
        return False

    # Check positive semidefiniteness using eigenvalues
    try:
        eigenvals = np.linalg.eigvals(matrix)
        # Matrix is positive semidefinite if all eigenvalues are non-negative
        # Using small tolerance to account for numerical errors
        if not np.all(eigenvals > -1e-8):
            return False
    except np.linalg.LinAlgError:
        return False

    return True


def nearest_correlation_matrix(matrix, *, weights=None, eps=1e-6, verbose=False):
    """Returns the positive definite correlation matrix nearest to `matrix`,
    weighted elementwise by `weights`.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix that we want to find the nearest positive definite
        correlation matrix to. A square 2-dimensional NumPy ndarray.
    weights : np.ndarray or None, optional
        An elementwise weighting matrix. A square 2-dimensional NumPy ndarray
        that must have the same shape as `matrix`. The default is None.
    eps : float, optional
        Tolerance for the optimization solver and minimum eigenvalue threshold.
        The result will have all eigenvalues > eps. The default is 1e-6.
    verbose : bool, optional
        Whether to print information from the solver. The default is False.

    Returns
    -------
    np.ndarray
        The positive definite correlation matrix that is nearest to the input
        matrix. All eigenvalues will be strictly positive.

    Notes
    -----
    This function implements equation (3) in the paper "An Augmented Lagrangian
    Dual Approach for the H-Weighted Nearest Correlation Matrix Problem" by
    Houduo Qi and Defeng Sun, with an additional constraint to ensure the
    result is positive definite (not just positive semidefinite).
        http://www.personal.soton.ac.uk/hdqi/REPORTS/Cor_matrix_H.pdf
    Another useful link is:
        https://nhigham.com/2020/04/14/what-is-a-correlation-matrix/

    The algorithm finds a matrix that is:
    1. A valid correlation matrix (diagonal elements = 1)
    2. Positive definite (all eigenvalues > eps)
    3. Closest to the input matrix in weighted Frobenius norm

    Examples
    --------
    >>> X = np.array([[1, 1, 0],
    ...               [1, 1, 1],
    ...               [0, 1, 1]])
    >>> nearest_correlation_matrix(X)
    array([[1.        , 0.76068..., 0.15729...],
           [0.76068..., 1.        , 0.76068...],
           [0.15729..., 0.76068..., 1.        ]])
    >>> H = np.array([[1,   0.5, 0.1],
    ...               [0.5,   1, 0.5],
    ...               [0.1, 0.5, 1]])
    >>> nearest_correlation_matrix(X, weights=H)
    array([[1.        , 0.94171..., 0.77365...],
           [0.94171..., 1.        , 0.94171...],
           [0.77365..., 0.94171..., 1.        ]])
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input argument `matrix` must be np.ndarray.")
    if not matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        raise ValueError("Input argument `matrix` must be square.")

    # Switch to notation used in the paper
    G = matrix.copy()
    H = np.ones_like(G) if weights is None else weights

    if not isinstance(H, np.ndarray):
        raise TypeError("Input argument `weights` must be np.ndarray.")
    if not (H.shape == G.shape):
        raise ValueError("Argument `weights` must have same shape as `matrix`.")

    # To constrain Y to be Positive Semidefinite (PSD), you need to
    # either set PSD=True here, or add the special constraint 'Y >> 0'. See:
    # https://www.cvxpy.org/tutorial/constraints/index.html#semidefinite-matrices
    X = cp.Variable(shape=G.shape, PSD=True)

    # Objective and constraints for minimizing the weighted frobenius norm.
    # This is equation (3) in the paper. We add (X - eps * I) >> 0 as an extra
    # constraint to ensure the result is positive definite (all eigenvalues > eps).
    # This constraint makes X - eps*I positive semidefinite, which means
    # all eigenvalues of X are > eps, ensuring positive definiteness.
    objective = cp.norm(cp.multiply(H, X - G), "fro")
    eps_identity = (eps / G.shape[0]) * 10
    constraints = [cp.diag(X) == 1.0, (X - eps_identity * np.eye(G.shape[0])) >> 0]

    # For solver options, see:
    # https://www.cvxpy.org/tutorial/solvers/index.html#setting-solver-options
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver="SCS", verbose=verbose, eps=eps)
    X = X.value.copy()  # Copy over solution

    # We might get small eigenvalues due to numerics. Attempt to fix this by
    # recursively calling the solver with smaller values of epsilon. This is
    # an extra fail-safe that ensures we maintain positive definiteness.
    is_symmetric = np.allclose(X, X.T)
    is_PD = np.linalg.eig(X)[0].min() > 0  # Check for positive definiteness
    if not (is_symmetric and is_PD) and (eps > 1e-14):
        if verbose:
            print(f"Recursively calling solver with eps := {eps} / 10")
        return nearest_correlation_matrix(G, weights=H, eps=eps / 10, verbose=verbose)

    return X


class DesignMatrix:
    """Class for design matrix in FMU. Can contain a onebyone design
    or a full montecarlo design.

    Attributes:
        designvalues (pd.DataFrame): design matrix on standard fmu format
            contains columns 'REAL' (realization number), and if a onebyone
            design, also columns 'SENSNAME' and 'SENSCASE'
        defaultvalues (OrderedDict): default values for design
        backgroundvalues (pd.DataFrame): Used when background parameters are
            not constant. Either a set is sampled from specified distributions
            or they are read from a file.
    """

    def __init__(self):
        """
        Placeholders for:
        designvalues: dataframe with parameters that varies
        defaultvalues: dictionary of default/base case values
        backgroundvalues: dataframe with background parameters
        seedvalues: list of seed values
        """
        self.designvalues = pd.DataFrame(columns=["REAL"])
        self.defaultvalues = OrderedDict()
        self.backgroundvalues = None
        self.seedvalues = None

    def reset(self):
        """Resets DesignMatrix to empty. Necessary iin case method generate
        is used several times for same instance of DesignMatrix"""
        self.designvalues = pd.DataFrame(columns=["REAL"])
        self.defaultvalues = OrderedDict()
        self.backgroundvalues = None
        self.seedvalues = None

    def generate(self, inputdict):
        """Generating design matrix from input dictionary in specific
        format. Adding default values and background values if existing.
        Looping through sensitivities and adding them to designvalues.

        Args:
            inputdict (OrderedDict): input parameters for design
        """

        if inputdict["designtype"] != "onebyone":
            raise ValueError(
                "Generation of DesignMatrix only "
                "implemented for type onebyone. "
                "In general_input designtype is "
                "set to {}".format(inputdict["designtype"])
            )

        self.reset()  # Emptying if regenerating matrix

        rng = np.random.RandomState(seed=inputdict.get("distribution_seed"))

        self.defaultvalues = inputdict["defaultvalues"]

        max_reals = _find_max_realisations(inputdict)

        # Reading or generating rms seed values
        if "seeds" in inputdict:
            self.add_seeds(inputdict["seeds"], max_reals)

        # If background values used - read or generate
        if "background" in inputdict:
            self.add_background(inputdict["background"], max_reals, rng)

        self.designvalues["SENSNAME"] = None
        self.designvalues["SENSCASE"] = None
        current_real_index = 0
        for key in inputdict["sensitivities"]:
            sens = inputdict["sensitivities"][key]
            numreal = sens["numreal"] if "numreal" in sens else inputdict["repeats"]
            if sens["senstype"] == "ref":
                sensitivity = SingleRealisationReference(key)
                sensitivity.generate(
                    range(current_real_index, current_real_index + numreal)
                )
                current_real_index += numreal
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "background":
                sensitivity = BackgroundSensitivity(key)
                sensitivity.generate(
                    range(current_real_index, current_real_index + numreal)
                )
                current_real_index += numreal
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "seed":
                if self.seedvalues is None:
                    raise ValueError(
                        "No seed values available to use for seed sensitivity"
                    )
                sensitivity = SeedSensitivity(key)
                sensitivity.generate(
                    range(current_real_index, current_real_index + numreal),
                    sens["seedname"],
                    self.seedvalues,
                    sens["parameters"],
                )
                current_real_index += numreal
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "scenario":
                sensitivity = ScenarioSensitivity(key)
                for casekey in sens["cases"]:
                    case = sens["cases"][casekey]
                    temp_case = ScenarioSensitivityCase(casekey)
                    temp_case.generate(
                        range(current_real_index, current_real_index + numreal),
                        case,
                        self.seedvalues,
                    )
                    sensitivity.add_case(temp_case)
                    current_real_index += numreal
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "dist":
                sensitivity = MonteCarloSensitivity(key)
                sensitivity.generate(
                    range(current_real_index, current_real_index + numreal),
                    sens["parameters"],
                    self.seedvalues,
                    sens["correlations"],
                    rng=rng,
                )
                current_real_index += numreal
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "extern":
                sensitivity = ExternSensitivity(key)
                sensitivity.generate(
                    range(current_real_index, current_real_index + numreal),
                    sens["extern_file"],
                    sens["parameters"],
                    self.seedvalues,
                )
                current_real_index += numreal
                self._add_sensitivity(sensitivity)
            print("Added sensitivity :", sensitivity.sensname)
        if "background" in inputdict:
            self._fill_with_background_values()
        self._fill_with_defaultvalues()

        if "dependencies" in inputdict:
            self._fill_derived_params(inputdict["dependencies"])

        if "decimals" in inputdict:
            self._set_decimals(inputdict["decimals"])
        # Re-order columns
        start_cols = ["REAL", "SENSNAME", "SENSCASE", "RMS_SEED"]
        self.designvalues = self.designvalues[
            [col for col in start_cols if col in self.designvalues]
            + [col for col in self.designvalues if col not in start_cols]
        ]

    def to_xlsx(
        self, filename, designsheet="DesignSheet01", defaultsheet="DefaultValues"
    ):
        """Writing design matrix to excel workfbook on standard fmu format
        to be used in FMU/ERT by DESIGN2PARAMS and DESIGN_KW

        Args:
            filename (str): output filename (extension .xlsx)
            designsheet (str): name of excel sheet containing design matrix
                (optional, defaults to 'DesignSheet01')
            defaultsheet (str): name of excel sheet containing default
                values (optional, defaults to 'DefaultValues')
        """
        # pylint: disable=abstract-class-instantiated
        basename = Path(filename).stem
        extension = Path(filename).suffix
        if extension != ".xlsx":
            filename = basename + ".xlsx"
            print(
                "Warning: Output filename did not have extension .xlsx "
                "but the export format is Excel .xlsx .  "
                f"Changing outputname to {filename}"
            )

        xlsxwriter = pd.ExcelWriter(filename, engine="openpyxl")
        self.designvalues.to_excel(
            xlsxwriter, sheet_name=designsheet, index=False, header=True
        )
        # Default values from OrderdDictionay to pandas dataframe
        defaults = pd.DataFrame(columns=["defaultparameters", "defaultvalue"])
        defaults["defaultparameters"] = self.defaultvalues.keys()
        defaults["defaultvalue"] = self.defaultvalues.values()
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
        if seeds in (None, "None"):
            self.seedvalues = None
            print("seeds is set to None in general_input")
        elif seeds and seeds.lower() == "default":
            self.seedvalues = [item + 1000 for item in range(max_reals)]
        elif seeds and Path(seeds).is_file():
            self.seedvalues = _seeds_from_extern(seeds, max_reals)
        else:
            raise ValueError(
                "Valid choices for seeds are None, "
                '"default" or an existing filename. '
                "Neither was found in this case. seeds "
                f"had been specified as {seeds} ."
            )

    def add_background(
        self, back_dict: OrderedDict, max_values: int, rng: np.random.RandomState
    ) -> None:
        """Adding background as specified in dictionary.
        Either from external file or from distributions in background
        dictionary

        Args:
            back_dict (OrderedDict): how to generate background values
            max_values (int): number of background values to generate
            rng (numpy.random.RandomState): Random number generator instance
        """
        if back_dict is None:
            self.backgroundvalues = None
        elif "extern" in back_dict:
            self.backgroundvalues = _parameters_from_extern(back_dict["extern"])
        elif "parameters" in back_dict:
            self._add_dist_background(back_dict, max_values, rng)

    def background_to_excel(self, filename, backgroundsheet="Background"):
        """Writing background values to an Excel spreadsheet

        Args:
            filename (str): output filename (extension .xlsx)
            backgroundsheet (str): name of excel sheet
        """
        # pylint: disable=abstract-class-instantiated
        xlsxwriter = pd.ExcelWriter(filename, engine="openpyxl")
        self.backgroundvalues.to_excel(
            xlsxwriter, sheet_name=backgroundsheet, index=False, header=True
        )
        xlsxwriter.close()
        print(f"Backgroundvalues written to {filename}")

    def _add_sensitivity(self, sensitivity):
        """Adding a sensitivity to the design

        Args:
            sensitivity of class Scenario, MonteCarlo or Extern
        """
        existing_values = self.designvalues.copy()
        self.designvalues = pd.concat([existing_values, sensitivity.sensvalues])

    def _fill_with_background_values(self):
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

    def _fill_with_defaultvalues(self):
        """Filling NaNs with default values"""
        for key in self.designvalues:
            if key in self.defaultvalues:
                self.designvalues[key] = self.designvalues[key].fillna(
                    self.defaultvalues[key]
                )
            elif key not in ["REAL", "SENSNAME", "SENSCASE", "RMS_SEED"]:
                raise LookupError(f"No defaultvalues given for parameter {key} ")

    def _fill_derived_params(self, depend_dict):
        for from_param in depend_dict:
            if from_param in self.designvalues:
                for param in depend_dict[from_param]["to_params"]:
                    self.designvalues[param] = np.nan
                    for index in range(len(depend_dict[from_param]["from_values"])):
                        fill_these = depend_dict[from_param]["from_values"][index]
                        fill_data = depend_dict[from_param]["to_params"][param][index]
                        self.designvalues[param] = self.designvalues[param].mask(
                            self.designvalues[from_param] == fill_these, fill_data
                        )
                    if self.designvalues[param].isnull().any():
                        raise ValueError(
                            f"Column for derived parameter {param} "
                            "contains NaN. Check input "
                            "defining dependencies. "
                            "Could be Wrong values or that "
                            "values for input variable  in "
                            "dependencies sheet "
                            "should be specified as strings."
                        )

    def _add_dist_background(
        self, back_dict: OrderedDict, numreal: int, rng: np.random.RandomState
    ) -> None:
        """Drawing background values from distributions
        specified in dictionary

        Args:
            back_dict (OrderedDict): parameters and distributions
            numreal (int): Number of samples to generate
            rng (numpy.random.RandomState): Random number generator instance
        """
        assert isinstance(
            numreal, int
        ), f"numreal must be an integer, got {type(numreal)} with value {numreal}"
        mc_background = MonteCarloSensitivity("background")
        mc_background.generate(
            range(numreal),
            back_dict["parameters"],
            "None",
            back_dict["correlations"],
            rng,
        )
        mc_backgroundvalues = mc_background.sensvalues

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

    def _set_decimals(self, dict_decimals):
        """Rounding to specified number of decimals

        Args:
            dict_decimals (dictionary): (key, value)s are (param, decimals)
        """
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


class SeedSensitivity:
    """
    A seed sensitivity is normally the reference for one by one sensitivities
    It contains a list of seeds to be repeated for each sensitivity
    The parameter name is hardcoded to RMS_SEED
    It will be assigned the sensname 'p10_p90' which will be written to
    the SENSCASE column in the output.

    Attributes:
        sensname (str): name of sensitivity
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    # pylint: disable=too-few-public-methods
    def __init__(self, sensname):
        """Initiate method.

        Args:
            sensname (str): Name of sensitivity. Defines SENSNAME in design matrix.
        """
        self.sensname = sensname
        self.sensvalues = None

    def generate(self, realnums, seedname, seedvalues, parameters):
        """Generates parameter values for a seed sensitivity

        Args:
            realnums (list): list of integers with realization numbers
            seedname (str): name of seed parameter to add
            seedvalues (list): list of integer seedvalues
            parameters (OrderedDict): parameter names and
                distributions or values.
        """
        assert isinstance(
            seedvalues, list
        ), f"seedvalues must be a list, got {seedvalues}"

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


class SingleRealisationReference:
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

    # pylint: disable=too-few-public-methods
    def __init__(self, sensname):
        """Initiate.

        Args:
            sensname (str): Name of sensitivity. Defines SENSNAME in design matrix.
        """
        self.sensname = sensname
        self.sensvalues = None

    def generate(self, realnums):
        """Generates realisation number only

        Args:
            realnums (list): list of integers with realization numbers
        """
        self.sensvalues = pd.DataFrame(index=realnums)
        self.sensvalues["REAL"] = realnums
        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "ref"


class BackgroundSensitivity:
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

    # pylint: disable=too-few-public-methods
    def __init__(self, sensname):
        """Initiate

        Args:
            sensname (str): Name of sensitivity. Defines SENSNAME in design matrix.
        """
        self.sensname = sensname
        self.sensvalues = None

    def generate(self, realnums):
        """Generates realisation number only

        Args:
            realnums (list): list of integers with realization numbers
        """
        self.sensvalues = pd.DataFrame(index=realnums)
        self.sensvalues["REAL"] = realnums
        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"


class ScenarioSensitivity:
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

    # pylint: disable=too-few-public-methods
    def __init__(self, sensname):
        """
        Args:
            sensname (str): Name of sensitivity.
                Equals SENSNAME in design matrix
        """
        self.sensname = sensname
        self.case1 = None
        self.case2 = None
        self.sensvalues = None

    def add_case(self, senscase):
        """
        Adds a ScenarioSensitivityCase instance
        to a ScenarioSensitivity object.

        Args:
            senscase (ScenarioSensitivityCase):
                Equals SENSCASE in design matrix.
        """
        if self.case1 is not None:  # Case 1 has been read, this is case2
            if "REAL" in senscase.casevalues and "SENSCASE" in senscase.casevalues:
                self.case2 = senscase
                senscase.casevalues["SENSNAME"] = self.sensname
                self.sensvalues = pd.concat(
                    [self.sensvalues, senscase.casevalues], sort=True
                )
        elif "REAL" in senscase.casevalues and "SENSCASE" in senscase.casevalues:
            self.case1 = senscase
            self.sensvalues = senscase.casevalues.copy()
            self.sensvalues["SENSNAME"] = self.sensname


class ScenarioSensitivityCase:
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
        casename (str): name of the sensitivity case,
            equals SENSCASE in design matrix.
        casevalues (pd.DataFrame): parameters and values
            for the sensitivity with realisation numbers as index.

    """

    # pylint: disable=too-few-public-methods
    def __init__(self, casename):
        self.casename = casename
        self.casevalues = None

    def generate(self, realnums, parameters, seedvalues):
        """Generate casevalues for the ScenarioSensitivityCase

        Args:
            realnums (list): list of realizaton numbers for the case
            parameters (OrderedDict):
                dictionary with parameter names and values
            seeds (str): default or None
        """

        self.casevalues = pd.DataFrame(columns=parameters.keys(), index=realnums)
        for key in parameters:
            self.casevalues[key] = parameters[key]
        self.casevalues["REAL"] = realnums
        self.casevalues["SENSCASE"] = self.casename

        if seedvalues:
            self.casevalues["RMS_SEED"] = seedvalues[: len(realnums)]


class MonteCarloSensitivity:
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

    # pylint: disable=too-few-public-methods
    def __init__(self, sensname):
        self.sensname = sensname
        self.sensvalues = None

    def _draw_uncorrelated_values(
        self, param_name, dist_name, dist_params, numreals, rng
    ):
        try:
            return design_dist.draw_values(
                dist_name.lower(), dist_params, numreals, rng
            )
        except ValueError as error:
            raise ValueError(
                f"Problem in sensitivity with sensname {self.sensname} "
                f"for parameter {param_name}: {error.args[0]}"
            ) from error

    def generate(self, realnums, parameters, seedvalues, corrdict, rng):
        """Generates parameter values by drawing from defined distributions.

        Args:
            realnums (range): range object containing realization numbers
            parameters (OrderedDict): dictionary of parameters and distributions
            seeds (str): default or None
            corrdict (OrderedDict): Configuration for correlated parameters. Contains:
                - 'inputfile': Path to Excel file with correlation matrices
                - 'sheetnames': List of sheet names, where each sheet contains a correlation matrix
                If None, parameters are treated as uncorrelated.
        """
        self.sensvalues = pd.DataFrame(columns=parameters.keys(), index=realnums)
        numreals = len(realnums)

        if corrdict is None:
            for key in parameters:
                self.sensvalues[key] = self._draw_uncorrelated_values(
                    key, parameters[key][0], parameters[key][1], numreals, rng
                )
        else:  # Some or all parameters are correlated
            df_params = pd.DataFrame.from_dict(
                parameters,
                orient="index",
                columns=["dist_name", "dist_params", "corr_sheet"],
            )
            df_params["corr_sheet"] = df_params["corr_sheet"].fillna("nocorr")
            df_params.reset_index(inplace=True)
            df_params.rename(columns={"index": "param_name"}, inplace=True)
            corr_groups = df_params.groupby("corr_sheet")

            for corr_group_name, corr_group in corr_groups:
                if corr_group_name != "nocorr":
                    if len(corr_group) == 1:
                        _printwarning(corr_group_name)
                    df_correlations = design_dist.read_correlations(
                        corrdict["inputfile"], corr_group_name
                    )
                    multivariate_parameters = df_correlations.index.values
                    correlations = df_correlations.values

                    # Make correlation matrix symmetric by copying lower triangular part
                    correlations = np.triu(correlations.T, k=1) + np.tril(correlations)

                    if not is_consistent_correlation_matrix(correlations):
                        print("\nWarning: Correlation matrix is not consistent")
                        print("Requirements:")
                        print("  - Ones on the diagonal")
                        print("  - Positive semi-definite matrix")
                        print("\nInput correlation matrix:")
                        with np.printoptions(
                            precision=2,
                            suppress=True,
                            formatter={"float": lambda x: f"{x:.2f}"},
                        ):
                            print(correlations)
                        correlations = nearest_correlation_matrix(
                            correlations, weights=None, eps=1e-6, verbose=False
                        )
                        print("\nAdjusted to nearest consistent correlation matrix:")
                        with np.printoptions(
                            precision=2,
                            suppress=True,
                            formatter={"float": lambda x: f"{x:.2f}"},
                        ):
                            print(correlations)
                        print()

                    sampler = qmc.LatinHypercube(
                        d=len(multivariate_parameters), seed=rng
                    )
                    lhs_samples = sampler.random(n=numreals)

                    iman_conover = ImanConover(correlation_matrix=correlations)
                    correlated_samples = iman_conover(lhs_samples)

                    # Transform uniform correlated samples to normal scores
                    normalscoresamples = scipy.stats.norm.ppf(correlated_samples)

                    for idx, row in corr_group.reset_index(drop=True).iterrows():
                        if row["param_name"] in multivariate_parameters:
                            self.sensvalues[row["param_name"]] = (
                                design_dist.draw_values(
                                    row["dist_name"].lower(),
                                    row["dist_params"],
                                    numreals,
                                    rng,
                                    normalscoresamples[:, idx],
                                )
                            )
                        else:
                            raise ValueError(
                                f"Parameter {row['param_name']} specified with correlation "
                                f"matrix {corr_group_name} but is not listed in that sheet"
                            )
                else:  # group nocorr where correlation matrix is not defined
                    for _, row in corr_group.iterrows():
                        self.sensvalues[row["param_name"]] = (
                            self._draw_uncorrelated_values(
                                row["param_name"],
                                row["dist_name"],
                                row["dist_params"],
                                numreals,
                                rng,
                            )
                        )

        if self.sensname != "background":
            self.sensvalues["REAL"] = realnums
            self.sensvalues["SENSNAME"] = self.sensname
            self.sensvalues["SENSCASE"] = "p10_p90"
            if "RMS_SEED" not in self.sensvalues and seedvalues:
                self.sensvalues["RMS_SEED"] = seedvalues[: len(realnums)]


class ExternSensitivity:
    """
    Used when reading parameter values from a file
    Assumed to be used with monte carlo type sensitivities and
    will hence write 'p10_p90' as SENSCASE in output designmatrix

    Attributes:
        sensname (str): Name of sensitivity.
            Defines SENSNAME in design matrix
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    # pylint: disable=too-few-public-methods
    def __init__(self, sensname):
        self.sensname = sensname
        self.sensvalues = None

    def generate(self, realnums, filename, parameters, seedvalues):
        """Reads parameter values for a monte carlo sensitivity
        from file

        Args:
            realnums (list): list of integers with realization numbers
            filename (str): path where to read values from
            parameters (list): list with parameter names
            seeds (str): default or None
        """
        self.sensvalues = pd.DataFrame(columns=parameters, index=realnums)
        extern_values = _parameters_from_extern(filename)
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


# Support functions used with several classes


def _parameters_from_extern(filename):
    """Read parameter values or background values
    from specified file. Format either Excel ('xlsx')
    or csv.

    Args:
        filename (str): path to file
    """
    if str(filename).endswith(".xlsx"):
        parameters = pd.read_excel(filename, engine="openpyxl")
        parameters.dropna(axis=0, how="all", inplace=True)
        parameters = parameters.loc[:, ~parameters.columns.str.contains("^Unnamed")]
    elif str(filename).endswith(".csv"):
        parameters = pd.read_csv(filename)
    else:
        raise ValueError(
            "External file with parameter values should "
            "be on Excel or csv format "
            "and end with .xlsx or .csv"
        )
    return parameters


def _seeds_from_extern(filename, max_reals):
    """Read parameter values or background values
    from specified file. Format either Excel ('xlsx')
    or csv.

    Args:
        filename (str): path to file
    """
    if str(filename).endswith(".xlsx"):
        df_seeds = pd.read_excel(filename, header=None, engine="openpyxl")
        df_seeds.dropna(axis=0, how="all", inplace=True)
        df_seeds.dropna(axis=1, how="all", inplace=True)
        seed_numbers = df_seeds[df_seeds.columns[0]].tolist()
    elif str(filename).endswith(".csv") or str(filename).endswith(".txt"):
        df_seeds = pd.read_csv(filename, header=None)
        seed_numbers = df_seeds[df_seeds.columns[0]].tolist()
    else:
        raise ValueError(
            "External file with seed values should "
            "be on Excel or csv format "
            "and end with .xlsx .csv or .txt"
        )

    if len(seed_numbers) < max_reals:
        print(
            "Provided number of seed values in external file {} "
            "is lower than the maximum number of realisations "
            "found for the design {}, and is for those "
            "sensitivities used repeatedly. "
        )
        seed_numbers = [
            seed_numbers[item % len(seed_numbers)] for item in range(max_reals)
        ]
    return seed_numbers


def _find_max_realisations(inputdict):
    """Finds the maximum number of realisations
    in a sensitivity case"""
    max_reals = inputdict["repeats"]
    for key in inputdict["sensitivities"]:
        sens = inputdict["sensitivities"][key]
        if "numreal" in sens:
            max_reals = max(sens["numreal"], max_reals)
    return max_reals


def _printwarning(corr_group_name):
    print(
        "#######################################################\n"
        "semeio.fmudesign Warning:                                     \n"
        "Using designinput sheets where "
        "corr_sheet is only specified for one parameter "
        "will cause non-correlated parameters .\n"
        f"ONLY ONE PARAMETER WAS SPECIFIED TO USE CORR_SHEET {corr_group_name}\n"
        "\n"
        "Note change in how correlated parameters are specified \n"
        "from fmudeisgn version 1.0.1 in August 2019 :\n"
        "Name of correlation sheet must be specified for each "
        "parameter in correlation matrix. \n"
        "This to enable use of several correlation sheets. "
        "This also means non-correlated parameters do not "
        "have to be included in correlation matrix. \n "
        "See documentation: \n"
        "https://equinor.github.io/fmu-tools/"
        "fmudesign.html#create-design-matrix-for-"
        "one-by-one-sensitivities\n"
        "\n"
        "####################################################\n"
    )
