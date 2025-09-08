"""Module for random sampling of parameter values from distributions."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import probabilit


def validate_params(distname: str, parameters: list[str]) -> list[float]:
    """Common parameter validation for all distributions.

    Example:
    >>> validate_params('normal', ['0', '-3.14', '1e10'])
    [0.0, -3.14, 10000000000.0]
    >>> validate_params('normal', ['inf'])
    Traceback (most recent call last):
        ...
    ValueError: Parameter 1 in distribution normal is not finite: inf
    """
    new_parameters: list[float] = []

    for i, parameter in enumerate(parameters):
        try:
            new_parameters.append(float(parameter))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Parameter {i + 1} in distribution {distname} not convertible to number: {parameter}"
            ) from e

        if not np.isfinite(new_parameters[i]):
            raise ValueError(
                f"Parameter {i + 1} in distribution {distname} is not finite: {parameter}"
            )

    return new_parameters


def to_probabilit(
    distname: str,
    dist_parameters: Sequence[str],
) -> npt.NDArray[Any] | list[str]:
    """
    Prepare scipy distributions with parameters
    Args:
        distname (str): distribution name 'normal', 'lognormal', 'triang',
        'uniform', 'logunif', 'discrete', 'pert'
        dist_parameters (list): list with parameters for distribution
    Returns:
        array with sampled values
    """

    # Special case for discrete
    if distname.lower().startswith("disc"):
        if len(dist_parameters) == 1:
            values = dist_parameters[0]
            values = [v.strip() for v in values.split(",")]
            return probabilit.DiscreteDistribution(values)
        else:
            values, probabilities = dist_parameters
            values = [v.strip() for v in values.split(",")]
            probabilities = [float(v.strip()) for v in probabilities.split(",")]
            return probabilit.DiscreteDistribution(values, probabilities)

    # Special case for constant
    if distname.lower().startswith("const"):
        return probabilit.Constant(dist_parameters[0])

    # Convert parameters
    parameters: list[float] = validate_params(
        distname=distname, parameters=list(dist_parameters)
    )

    # Normal distribution
    if distname.lower().startswith("norm"):
        if len(parameters) == 2:
            mean, stddev = parameters
            return probabilit.Distribution("norm", loc=mean, scale=stddev)
        elif len(parameters) == 4:
            mean, stddev, low, high = parameters
            return probabilit.distributions.TruncatedNormal(
                loc=mean, scale=stddev, low=low, high=high
            )
        else:
            raise ValueError(f"Normal must have 2 or 4 parameters, got: {parameters}")

    elif distname.lower().startswith("logn"):
        if len(parameters) != 2:
            raise ValueError(f"Lognormal must have 2 parameters, got: {parameters}")
        mean, sigma = parameters
        return probabilit.distributions.Lognormal.from_log_params(mu=mean, sigma=sigma)
    elif distname.lower().startswith("unif"):
        if len(parameters) != 2:
            raise ValueError(f"Uniform must have 2 parameters, got: {parameters}")
        low, high = parameters
        return probabilit.distributions.Uniform(min=low, max=high)
    elif distname.lower().startswith("triang"):
        if len(parameters) != 3:
            raise ValueError(f"Triangular must have 3 parameters, got: {parameters}")
        low, mode, high = parameters
        return probabilit.distributions.Triangular(
            low=low, mode=mode, high=high, low_perc=0.0, high_perc=1.0
        )
    elif distname.lower().startswith("pert"):
        if len(parameters) == 3:
            low, mode, high = parameters
            return probabilit.distributions.PERT(minimum=low, mode=mode, maximum=high)
        elif len(parameters) == 4:
            low, mode, high, scale = parameters
            return probabilit.distributions.PERT(
                minimum=low, mode=mode, maximum=high, scale=scale
            )
        else:
            raise ValueError(f"PERT must have 3 or 4 parameters, got: {parameters}")
    elif distname.lower().startswith("logunif"):
        if len(parameters) != 2:
            raise ValueError(f"Loguniform must have 2 parameters, got: {parameters}")
        low, high = parameters
        return probabilit.Distribution("loguniform", a=low, b=high)
    else:
        raise ValueError(f"Distribution name {distname} is not implemented")


def is_number(teststring: str) -> bool:
    """Test if a string can be parsed as a float"""
    try:
        return not np.isnan(float(teststring))
    except ValueError:
        return False


def read_correlations(excel_filename: str | Path, corr_sheet: str) -> pd.DataFrame:
    """Reading correlation info for a
    monte carlo sensitivity

    Args:
        excel_filename (Path): Path to Excel file containing correlation matrix
        corr_sheet (str): name of sheet containing correlation matrix

    Returns:
        pd.DataFrame: Dataframe with correlations, parameter names
            as column and index
    """
    if not str(excel_filename).endswith(".xlsx"):
        raise ValueError(
            "Correlation matrix filename should be on Excel format and end with .xlsx"
        )

    correlations = pd.read_excel(
        excel_filename, corr_sheet, index_col=0, engine="openpyxl"
    )
    correlations.dropna(axis=0, how="all", inplace=True)
    # Remove any 'Unnamed' columns that Excel/pandas may have automatically added.
    correlations = correlations.loc[:, ~correlations.columns.str.contains("^Unnamed")]

    return correlations
