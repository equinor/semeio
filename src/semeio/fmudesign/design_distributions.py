"""Module for random sampling of parameter values from distributions."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import probabilit  # type: ignore[import-untyped]
from scipy import stats


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


def quantiles_to_values(
    *,
    quantiles: npt.NDArray[Any],
    values: npt.NDArray[Any],
    probabilities: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    """Maps quantiles to values (which can be categorical or not).

    Assume values = [A, B, C], then probabilities = [1, 1, 1] = [1/3, 1/3, 1/3],
    first we bin the interval [0, 1) into segments matching the probabilities:
    - A: [0, 1/3)
    - B: [1/3, 2/3)
    - C: [2/3, 1)
    Then we map the quantiles into the ranges back onto the original values.

    Examples
    --------
    >>> values = np.array(["A", "B", "C"])
    >>> quantiles = np.array([0, 1/3, 0.5, 0.8])
    >>> quantiles_to_values(quantiles=quantiles, values=values)
    array(['A', 'B', 'B', 'C'], dtype='<U1')

    >>> probabilities = np.array([0.2, 0.3, 0.5])
    >>> quantiles = (np.arange(1, 10)) / 10
    >>> quantiles_to_values(quantiles=quantiles, values=values,
    ...                     probabilities=probabilities) # [0.1, 0.2, ...]
    array(['A', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], dtype='<U1')
    """
    quantiles = np.array(quantiles)
    values = np.array(values)

    # If no probabilities are given, assume equal
    if probabilities is None:
        probabilities = np.ones(len(values)) / len(values)

    if not np.isclose(np.sum(probabilities), 1.0):
        probabilities = probabilities / np.sum(probabilities)

    assert np.all(probabilities >= 0)

    # Create bin edges
    edges = np.cumsum([0, *list(probabilities)])
    # Map to bin indices, then back onto the original values
    bin_indices = np.digitize(quantiles, edges, right=False) - 1
    return values[bin_indices]


def to_probabilit(
    distname: str,
    dist_parameters: Sequence[str],
) -> probabilit.modeling.AbstractDistribution:
    """
    Prepare scipy distributions with parameters
    Args:
        distname (str): distribution name 'normal', 'lognormal', 'triang',
        'uniform', 'logunif', 'discrete', 'pert', 'beta'
        dist_parameters (list): list with parameters for distribution
    Returns:
        array with sampled values
    """

    # A discrete variable is a distribution over categoricals, e.g. ('A', 'B', 'C')
    # with weights (0.5, 0.3, 0.2). The way we deal with them is that we sample uniform
    # values, then assign the interval [0, 0.5) to A, [0.5, 0.8) to B and [0.8, 1) to C.
    # This means that we can "correlate" these variables, in the sense that if their
    # underlying Uniforms are correlated, then the categorical values will often match too.
    # To accomplish all of this we assign _values and _probabilities to the distribution
    # instances below. This "correlation" only exists in a narrow specific sense of course.

    if distname.lower().startswith("disc"):
        if len(dist_parameters) == 1:
            values_str = dist_parameters[0]
            values = [v.strip() for v in values_str.split(",")]
            distr = probabilit.Distribution("uniform")
            distr._values = np.array(values)
            return distr
        else:
            values_str, probabilities_str = dist_parameters
            values = [v.strip() for v in values_str.split(",")]
            probabilities = [float(v.strip()) for v in probabilities_str.split(",")]
            distr = probabilit.Distribution("uniform")
            distr._values = np.array(values)
            distr._probabilities = np.array(probabilities)
            return distr

    # Special case for constant
    if distname.lower().startswith("const"):
        return probabilit.Constant(dist_parameters[0])

    # Convert parameters
    parameters: list[float] = validate_params(
        distname=distname, parameters=list(dist_parameters)
    )

    # Normal distribution
    if distname.lower().startswith("norm"):
        if len(parameters) not in (2, 4):
            raise ValueError(f"Normal must have 2 or 4 parameters, got: {parameters}")
        if distname.lower().startswith("normal_p10_p90"):
            p10, p90 = parameters[:2]
            # We use the equations
            # p10 = mu - z*sigma and p90 = mu + z*sigma
            # to find mu and sigma
            z_score = stats.norm.ppf(0.9)
            mean = (p10 + p90) / 2
            stddev = (p90 - p10) / (2 * z_score)
        else:
            mean, stddev = parameters[:2]
        if len(parameters) == 2:
            return probabilit.distributions.Normal(mean, stddev)
        elif len(parameters) == 4:
            low, high = parameters[2:]
            return probabilit.distributions.TruncatedNormal(
                mean, stddev, low=low, high=high
            )

    elif distname.lower().startswith("logn"):
        if len(parameters) != 2:
            raise ValueError(f"Lognormal must have 2 parameters, got: {parameters}")
        mean, sigma = parameters
        return probabilit.distributions.Lognormal.from_log_params(mu=mean, sigma=sigma)
    elif distname.lower().startswith("unif"):
        if len(parameters) != 2:
            raise ValueError(f"Uniform must have 2 parameters, got: {parameters}")
        if distname.lower().startswith("uniform_p10_p90"):
            p10, p90 = parameters
            length = (p90 - p10) / 0.8
            low = p10 - 0.1 * length
            high = p90 + 0.1 * length
        else:
            low, high = parameters
        return probabilit.distributions.Uniform(minimum=low, maximum=high)
    elif distname.lower().startswith("triang"):
        if len(parameters) != 3:
            raise ValueError(f"Triangular must have 3 parameters, got: {parameters}")
        low, mode, high = parameters
        if distname.lower().startswith("triangular_p10_p90"):
            return probabilit.distributions.Triangular(
                low=low, mode=mode, high=high, low_perc=0.1, high_perc=0.9
            )
        else:
            return probabilit.distributions.Triangular(
                low=low, mode=mode, high=high, low_perc=0.0, high_perc=1.0
            )
    elif distname.lower().startswith("beta"):
        if len(parameters) == 2:
            a, b = parameters
            # Defaults to probabilit.Distribution("beta", a=a, b=b, loc=0, scale=1)
            return probabilit.Distribution("beta", a=a, b=b)
        if len(parameters) == 4:
            a, b, low, high = parameters
            loc = low
            scale = high - low
            return probabilit.Distribution("beta", a=a, b=b, loc=loc, scale=scale)
        else:
            raise ValueError(f"Beta must have 2 or 4 parameters, got: {parameters}")
    elif distname.lower().startswith("pert"):
        if len(parameters) not in (3, 4):
            raise ValueError(f"PERT must have 3 or 4 parameters, got: {parameters}")
        if distname.lower().startswith("pert_p10_p90"):
            if len(parameters) == 3:
                low, mode, high = parameters
                return probabilit.distributions.PERT(
                    low=low, mode=mode, high=high, low_perc=0.1, high_perc=0.9
                )
            elif len(parameters) == 4:
                low, mode, high, scale = parameters
                return probabilit.distributions.PERT(
                    low=low,
                    mode=mode,
                    high=high,
                    low_perc=0.1,
                    high_perc=0.9,
                    gamma=scale,
                )
        elif len(parameters) == 3:
            low, mode, high = parameters
            return probabilit.distributions.PERT(
                low=low, mode=mode, high=high, low_perc=0.0, high_perc=1.0
            )
        elif len(parameters) == 4:
            low, mode, high, scale = parameters
            return probabilit.distributions.PERT(
                low=low,
                mode=mode,
                high=high,
                low_perc=0.0,
                high_perc=1.0,
                gamma=scale,
            )
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


def read_correlations(excel_filename: str, corr_sheet: str) -> pd.DataFrame:
    """Read a correlation matrix from an Excel sheet.

    The sheet must have rows/columns with variable names. They must match.
    The upper-triangular part must be empty strings. The lower triangular part
    must be specified.

    Args:
        excel_filename (str): name of Excel file containing correlation matrix
        corr_sheet (str): name of sheet containing correlation matrix

    Returns:
        pd.DataFrame: Dataframe with correlations, parameter names
            as column and index
    """
    if not str(excel_filename).endswith(".xlsx"):
        raise ValueError(
            "Correlation matrix filename should be on Excel format and end with .xlsx"
        )

    correlations = (
        pd.read_excel(
            excel_filename, sheet_name=corr_sheet, index_col=0, engine="openpyxl"
        )
        .dropna(axis=0, how="all")
        # Remove any 'Unnamed' columns that Excel/pandas may have automatically added.
        .loc[:, lambda df: ~df.columns.str.contains("^Unnamed")]
        # Remove whitespace
        .rename(columns=str.strip)
        .rename(index=str.strip)
    )

    if list(correlations.index) != list(correlations.columns):
        msg = f"Mismatch between column and index in correlation matrix sheet: {corr_sheet!r}\n"
        msg += f"Column: {correlations.columns.tolist()}\n"
        msg += f"Index : {correlations.index.tolist()}\n"
        msg += f"These values must match exactly. Please fix sheet {corr_sheet!r} in file {excel_filename!r}."
        raise ValueError(msg)

    upper_idx = np.triu_indices_from(correlations.values, k=1)
    lower_idx = np.tril_indices_from(correlations.values, k=0)  # Include diag
    lower_entries = correlations.values[lower_idx]

    if not np.all(np.isnan(correlations.values[upper_idx])):
        raise ValueError(
            f"All upper-triangular elements in matrix in corr sheet {corr_sheet} must be blank."
        )

    if not np.all(np.isfinite(lower_entries)):
        raise ValueError(
            f"All lower-triangular elements in matrix in corr sheet {corr_sheet} must be specified."
        )

    if np.any((lower_entries < -1) | (lower_entries > 1)):
        raise ValueError(
            f"All lower-triangular elements in matrix in corr sheet {corr_sheet} must be between 0 and 1."
        )

    # Upper triangular part is NaN. Fill it with 0. Copy lower triang over
    correlations = correlations.astype(float).fillna(0)
    mat = correlations.values + correlations.values.T
    np.fill_diagonal(mat, 1.0)
    correlations.loc[:] = mat
    return correlations
