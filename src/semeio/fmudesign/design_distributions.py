"""Module for random sampling of parameter values from
distributions. For use in generation of design matrices
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
from probabilit.distributions import Lognormal, Normal, Triangular, Uniform
from scipy.stats import qmc


def parse_and_validate_normal_params(
    dist_params: Sequence[int | str | float],
) -> tuple[float, ...]:
    if len(dist_params) not in [2, 4]:
        raise ValueError(
            "Normal distribution must have 2 parameters"
            " or 4 for a truncated normal, "
            f"but had {len(dist_params)} parameters."
        )
    try:
        params = [float(p) for p in dist_params]
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"All parameters must be convertible to numbers. Got: {dist_params}"
        ) from e

    if np.any(np.isnan(params)):
        raise ValueError(f"Parameters cannot be NaN. Got: {params}")

    if params[1] < 0:
        raise ValueError(
            f"Stddev for normal distribution must be >= 0. Got: {params[1]}"
        )

    if len(params) == 4 and params[2] >= params[3]:
        raise ValueError(
            "For truncated normal distribution, "
            "lower bound must be less than upper bound, "
            f"but got [{params[2]}, {params[3]}]."
        )
    return tuple(params)


def parse_and_validate_lognormal_params(
    dist_params: Sequence[int | str | float],
) -> tuple[float, float]:
    if len(dist_params) != 2:
        raise ValueError(
            "Lognormal distribution must have 2 parameters, "
            f"but had {len(dist_params)} parameters."
        )
    try:
        mean, stddev = [float(p) for p in dist_params]
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"All parameters must be convertible to numbers. Got: {dist_params}"
        ) from e

    if np.any(np.isnan([mean, stddev])):
        raise ValueError(f"Parameters cannot be NaN. Got: {[mean, stddev]}")

    if stddev < 0:
        raise ValueError(
            f"Stddev for lognormal distribution must be >= 0. Got: {stddev}"
        )
    return mean, stddev


def parse_and_validate_uniform_params(
    dist_params: Sequence[int | str | float],
) -> tuple[float, float]:
    if len(dist_params) != 2:
        raise ValueError(
            f"Uniform distribution requires exactly 2 parameters, got {len(dist_params)}"
        )
    try:
        low, high = [float(p) for p in dist_params]
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"All parameters must be convertible to numbers. Got: {dist_params}"
        ) from e
    if np.any(np.isnan([low, high])):
        raise ValueError(f"Parameters cannot be NaN. Got: low={low}, high={high}")
    if high < low:
        raise ValueError(f"Parameters must satisfy low < high, got [{low}, {high}]")
    return low, high


def parse_and_validate_triangular_params(
    dist_params: Sequence[int | str | float],
) -> tuple[float, float, float]:
    if len(dist_params) != 3:
        raise ValueError(
            f"Triangular distribution requires exactly 3 parameters, got {len(dist_params)}"
        )

    try:
        low, mode, high = [float(p) for p in dist_params]
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"All parameters must be convertible to numbers. Got: {dist_params}"
        ) from e

    if np.any(np.isnan([low, mode, high])):
        raise ValueError(
            f"Parameters cannot be NaN. Got: low={low}, mode={mode}, high={high}"
        )

    if not (low <= mode <= high):
        raise ValueError(
            f"Parameters must satisfy low <= mode <= high, got [{low}, {mode}, {high}]"
        )

    return low, mode, high


def parse_and_validate_pert_params(
    dist_params: Sequence[int | str | float],
) -> tuple[float, float, float, float]:
    if len(dist_params) not in [3, 4]:
        raise ValueError(
            f"PERT distribution must have 3 or 4 parameters, but had {len(dist_params)} parameters."
        )
    try:
        params = [float(p) for p in dist_params]
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"All parameters must be convertible to numbers. Got: {dist_params}"
        ) from e

    if np.any(np.isnan(params)):
        raise ValueError(f"Parameters cannot be NaN. Got: {params}")

    low, mode, high = params[0], params[1], params[2]
    if not (low <= mode <= high):
        raise ValueError(
            "For PERT distribution, parameters must satisfy low <= mode <= high, "
            f"but got low={low}, mode={mode}, high={high}."
        )

    scale = params[3] if len(params) == 4 else 4.0
    return low, mode, high, scale


def parse_and_validate_loguniform_params(
    dist_params: Sequence[int | str | float],
) -> tuple[float, float]:
    if len(dist_params) != 2:
        raise ValueError(
            f"Loguniform distribution requires exactly 2 parameters, got {len(dist_params)}"
        )
    try:
        low, high = [float(p) for p in dist_params]
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"All parameters must be convertible to numbers. Got: {dist_params}"
        ) from e
    if np.any(np.isnan([low, high])):
        raise ValueError(f"Parameters cannot be NaN. Got: low={low}, high={high}")
    if low <= 0:
        raise ValueError(f"For loguniform distribution, low must be > 0, got {low}")
    if high < low:
        raise ValueError(f"Parameters must satisfy low <= high, got [{low}, {high}]")
    return low, high


def generate_stratified_samples(
    numreals: int, rng: np.random.Generator
) -> npt.NDArray[Any]:
    """Generate stratified samples in [0,1] by dividing the interval
    into equal-probability strata.

    This is equivalent to one-dimensional Latin Hypercube Sampling
    where the [0,1] interval is divided into numreals equal segments,
    and one random sample is drawn from each segment.

    Parameters:
        numreals: int, number of samples to generate
        rng: numpy.random.Generator, random number generator instance

    Returns:
        numpy.ndarray: Array of stratified samples in [0,1]
    """
    if numreals < 0:
        raise ValueError("numreal must be a positive integer")

    if numreals == 0:
        return np.array([])

    sampler = qmc.LatinHypercube(d=1, rng=rng)
    samples = sampler.random(n=numreals)
    return samples.flatten()


def draw_values_normal(
    dist_parameters: Sequence[str],
    numreals: int,
    rng: np.random.Generator,
    normalscoresamples: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    params = parse_and_validate_normal_params(dist_parameters)
    mean, stddev = params[0], params[1]

    if len(params) == 2:  # normal
        if normalscoresamples is not None:
            values = mean + normalscoresamples * stddev
        else:
            values = Normal(mean, stddev).sample(numreals, rng, method="lhs")

    else:  # truncated normal
        clip1, clip2 = params[2], params[3]
        low = (clip1 - mean) / stddev
        high = (clip2 - mean) / stddev

        if normalscoresamples is not None:
            values = scipy.stats.truncnorm.ppf(
                scipy.stats.norm.cdf(normalscoresamples),
                low,
                high,
                loc=mean,
                scale=stddev,
            )
        else:
            uniform_samples = generate_stratified_samples(numreals, rng)
            values = scipy.stats.truncnorm.ppf(
                uniform_samples.flatten(), low, high, loc=mean, scale=stddev
            )

    return values


def draw_values_lognormal(
    dist_parameters: Sequence[int | str | float],
    numreals: int,
    rng: np.random.Generator,
    normalscoresamples: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    """Draws values from lognormal distribution.
    Args:
        dist_parameters(list): [mu, sigma] for the logarithm of the variable
        numreals(int): number of realisations to draw
        rng: numpy.random.Generator instance
        normalscoresamples(list): samples for correlated parameters
    Returns:
        list of values
    """
    mean, sigma = parse_and_validate_lognormal_params(dist_parameters)
    if normalscoresamples is not None:
        values = scipy.stats.lognorm.ppf(
            scipy.stats.norm.cdf(normalscoresamples),
            s=sigma,
            loc=0,
            scale=np.exp(mean),
        )
    else:
        values = Lognormal.from_log_params(mean, sigma).sample(numreals, rng)
    return values


def draw_values_uniform(
    dist_parameters: Sequence[int | str | float],
    numreals: int,
    rng: np.random.Generator,
    normalscoresamples: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    """Draws values from uniform distribution.
    Args:
        dist_parameters(list): [minimum, maximum]
        numreals(int): number of realisations to draw
        rng: numpy.random.Generator instance
        normalscoresamples(list): samples for correlated parameters
    Returns:
        list of values
    """
    if numreals < 0:
        raise ValueError("numreal must be a positive integer")

    if numreals == 0:
        return np.array([])

    low, high = parse_and_validate_uniform_params(dist_parameters)

    uscale = high - low

    if normalscoresamples is not None:
        return scipy.stats.uniform.ppf(
            scipy.stats.norm.cdf(normalscoresamples), loc=low, scale=uscale
        )

    return Uniform(low, high).sample(numreals, rng)


def draw_values_triangular(
    dist_parameters: Sequence[str],
    numreals: int,
    rng: np.random.Generator,
    normalscoresamples: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    """Draws values from triangular distribution.
    Args:
        dist_parameters(list): [min, mode, max]
        numreals(int): number of realisations to draw
        rng: numpy.random.Generator instance
        normalscoresamples(list): samples for correlated parameters
    Returns:
        list of values
    """
    low, mode, high = parse_and_validate_triangular_params(dist_parameters)

    dist_scale = high - low

    if dist_scale == 0:
        raise ValueError(
            f"Invalid triangular distribution: minimum ({low}) and maximum ({high}) cannot be equal"
        )

    shape = (mode - low) / dist_scale

    if normalscoresamples is not None:
        values = scipy.stats.triang.ppf(
            scipy.stats.norm.cdf(normalscoresamples),
            shape,
            loc=low,
            scale=dist_scale,
        )
    else:
        values = Triangular(low, mode, high, low_perc=0, high_perc=1).sample(
            numreals, rng
        )

    return values


def draw_values_pert(
    dist_parameters: Sequence[str],
    numreals: int,
    rng: np.random.Generator,
    normalscoresamples: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    """Draws values from pert distribution.
    Args:
        dist_parameters(list): [min, mode, max, scale]
        where scale is only specified
        for a 4 parameter pert distribution
        numreals(int): number of realisations to draw
        rng: numpy.random.Generator instance
        normalscoresamples(list): samples for correlated parameters
    Returns:
        list of values
    """
    low, mode, high, scale = parse_and_validate_pert_params(dist_parameters)

    if high == low:
        raise ValueError(
            f"Invalid PERT distribution: minimum ({low}) and maximum ({high}) cannot be equal"
        )

    muval = (low + high + scale * mode) / (scale + 2)
    if np.isclose(muval, mode):
        alpha1 = (scale / 2) + 1
    else:
        alpha1 = ((muval - low) * (2 * mode - low - high)) / (
            (mode - muval) * (high - low)
        )
    alpha2 = alpha1 * (high - muval) / (muval - low)

    if normalscoresamples is not None:
        uniform_samples = scipy.stats.norm.cdf(normalscoresamples)
    else:
        uniform_samples = generate_stratified_samples(numreals, rng)
    values = scipy.stats.beta.ppf(uniform_samples, alpha1, alpha2)

    # Scale the beta distribution to the desired range
    return values * (high - low) + low


def draw_values_loguniform(
    dist_parameters: Sequence[int | str | float],
    numreals: int,
    rng: np.random.Generator,
    normalscoresamples: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    """Draws values from loguniform distribution.
    Args:
        dist_parameters(list): [minimum, maximum]
        numreals(int): number of realisations to draw
        rng: numpy.random.Generator instance
        normalscoresamples(list): samples for correlated parameters
    Returns:
        list of values
    """
    low, high = parse_and_validate_loguniform_params(dist_parameters)

    if normalscoresamples is not None:
        values = scipy.stats.reciprocal.ppf(
            scipy.stats.norm.cdf(normalscoresamples), low, high
        )
    else:
        uniform_samples = generate_stratified_samples(numreals, rng)
        values = scipy.stats.reciprocal.ppf(uniform_samples, low, high)

    return values


def draw_values(
    distname: str,
    dist_parameters: Sequence[str],
    numreals: int,
    rng: np.random.Generator,
    normalscoresamples: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any] | list[str]:
    """
    Prepare scipy distributions with parameters
    Args:
        distname (str): distribution name 'normal', 'lognormal', 'triang',
        'uniform', 'logunif', 'discrete', 'pert'
        dist_parameters (list): list with parameters for distribution
        numreals (int): number of realizations to generate
        rng (numpy.random.Generator): random number generator instance
        normalscoresamples (array, optional): samples for correlated distributions
    Returns:
        scipy.stats distribution with parameters
    """
    if distname[0:4].lower() == "norm":
        values = draw_values_normal(dist_parameters, numreals, rng, normalscoresamples)
    elif distname[0:4].lower() == "logn":
        values = draw_values_lognormal(
            dist_parameters, numreals, rng, normalscoresamples
        )
    elif distname[0:4].lower() == "unif":
        values = draw_values_uniform(dist_parameters, numreals, rng, normalscoresamples)
    elif distname[0:6].lower() == "triang":
        values = draw_values_triangular(
            dist_parameters, numreals, rng, normalscoresamples
        )
    elif distname[0:4].lower() == "pert":
        values = draw_values_pert(dist_parameters, numreals, rng, normalscoresamples)
    elif distname[0:7].lower() == "logunif":
        values = draw_values_loguniform(
            dist_parameters, numreals, rng, normalscoresamples
        )
    elif distname[0:5].lower() == "const":
        if normalscoresamples is not None:
            raise ValueError(
                "Parameter with const distribution "
                "was defined in correlation matrix "
                "but const distribution cannot "
                "be used with correlation. "
            )
        values = np.array([dist_parameters[0]] * numreals)
    elif distname[0:4].lower() == "disc":
        status, result = sample_discrete(
            dist_parameters, numreals, rng, normalscoresamples
        )
        if status:
            values = result
        else:
            raise ValueError(result)
    else:
        raise ValueError(f"distribution name {distname} is not implemented")
    return values


def sample_discrete(
    dist_params: Sequence[str],
    numreals: int,
    rng: np.random.Generator,
    normalscoresamples: npt.NDArray[Any] | None = None,
) -> tuple[bool, npt.NDArray[Any]]:
    status = True
    outcomes = re.split(",", dist_params[0])
    outcomes = [item.strip() for item in outcomes]
    if numreals == 0:
        return status, np.array([])
    if numreals < 0:
        raise ValueError("numreal must be a positive integer")
    # Handle probability weights
    if len(dist_params) == 2:  # non uniform
        weights = re.split(",", dist_params[1])
        if len(outcomes) != len(weights):
            raise ValueError(
                "Number of weights for discrete distribution "
                "is not the same as number of values."
            )
        try:
            probabilities = [float(weight) for weight in weights]
        except ValueError as e:
            raise ValueError(
                "All weights must be valid floating point numbers. "
                f"Got weights: {weights}"
            ) from e
        # Just validate they're not negative
        if any(p < 0 for p in probabilities):
            raise ValueError("Weights cannot be negative")
        fractions = probabilities / np.sum(probabilities)
    elif len(dist_params) == 1:  # uniform
        fractions = [1.0 / len(outcomes)] * len(outcomes)
    else:
        raise ValueError("Wrong input for discrete distribution")
    # Handle correlation through normalscoresamples
    if normalscoresamples is not None:
        uniform_samples = scipy.stats.norm.cdf(normalscoresamples)
    else:
        uniform_samples = generate_stratified_samples(numreals, rng)
    cum_prob = np.cumsum(fractions)
    values = np.array([outcomes[np.searchsorted(cum_prob, s)] for s in uniform_samples])
    return status, values


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
