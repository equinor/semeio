"""Module for random sampling of parameter values from distributions."""

import dataclasses
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
from scipy.stats import qmc


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


class Distribution:
    def validate_params(self) -> None:
        """Common parameter validation for all distributions."""

        # Check that all parameters are numbers
        for param_name, param_value in self.__dict__.items():
            try:
                setattr(self, param_name, float(param_value))
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Parameter {param_name}={param_value} not convertible to number in: {self}"
                ) from e

        # Check low and high if distribution has these
        if hasattr(self, "low") and hasattr(self, "high"):
            if self.high <= self.low:
                raise ValueError(f"Must have high > low in: {self}")

            # Check mode if distribution has it
            if hasattr(self, "mode") and not (self.low <= self.mode <= self.high):
                raise ValueError(f"Must have low <= mode <= high in {self}")


@dataclasses.dataclass
class Normal(Distribution):
    """Create a normal distribution.

    Example:
    >>> rng = np.random.default_rng(42)
    >>> Normal(mean=0, stddev=2).sample(5, rng=rng)
    array([-0.42093622, -1.55927141, -3.9308858 ,  1.27522068,  1.74893971])
    >>> Normal(mean=0, stddev=2, low=10, high=5)
    Traceback (most recent call last):
        ...
    ValueError: Must have high > low in: Normal(mean=0.0, stddev=2.0, low=10.0, high=5.0)
    """

    mean: float = 0.0
    stddev: float = 1.0
    low: float = -np.inf
    high: float = np.inf

    def __post_init__(self) -> None:
        super().validate_params()
        if self.stddev < 0:
            raise ValueError(f"Must have positive stddev in: {self}")

    def sample(
        self,
        size: int,
        rng: np.random.Generator,
        normalscoresamples: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        # Scipy is parametrized wrt loc and scale
        low = (self.low - self.mean) / self.stddev
        high = (self.high - self.mean) / self.stddev

        if normalscoresamples is not None:
            return scipy.stats.truncnorm.ppf(
                scipy.stats.norm.cdf(normalscoresamples),
                low,
                high,
                loc=self.mean,
                scale=self.stddev,
            )

        uniform_samples = generate_stratified_samples(size, rng)
        return scipy.stats.truncnorm.ppf(
            uniform_samples.flatten(), low, high, loc=self.mean, scale=self.stddev
        )


@dataclasses.dataclass
class Lognormal(Distribution):
    mean: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        super().validate_params()
        if self.sigma <= 0:
            raise ValueError(f"Must have positive sigma in: {self}")

    def sample(
        self,
        size: int,
        rng: np.random.Generator,
        normalscoresamples: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        if normalscoresamples is not None:
            return scipy.stats.lognorm.ppf(
                scipy.stats.norm.cdf(normalscoresamples),
                s=self.sigma,
                loc=0,
                scale=np.exp(self.mean),
            )

        uniform_samples = generate_stratified_samples(size, rng)
        return scipy.stats.lognorm.ppf(
            uniform_samples, s=self.sigma, loc=0, scale=np.exp(self.mean)
        )


@dataclasses.dataclass
class Uniform(Distribution):
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        super().validate_params()

    def sample(
        self,
        size: int,
        rng: np.random.Generator,
        normalscoresamples: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        scale = self.high - self.low

        if normalscoresamples is not None:
            return scipy.stats.uniform.ppf(
                scipy.stats.norm.cdf(normalscoresamples), loc=self.low, scale=scale
            )

        uniform_samples = generate_stratified_samples(size, rng)
        return scipy.stats.uniform.ppf(uniform_samples, loc=self.low, scale=scale)


@dataclasses.dataclass
class Loguniform(Distribution):
    """Create a Loguniform random variable.

    Example:
    >>> rng = np.random.default_rng(0)
    >>> Loguniform(low=1, high=2).sample(5, rng=rng)
    array([1.75492879, 1.26289314, 1.03924188, 1.48955292, 1.64194378])
    >>> Loguniform(low=0, high=2).sample(5, rng=rng)
    Traceback (most recent call last):
      ...
    ValueError: Must have 0 < low < high in: Loguniform(low=0.0, high=2.0)
    """

    low: float = 1e-6
    high: float = 1.0

    def __post_init__(self) -> None:
        super().validate_params()
        if not (0 < self.low < self.high):
            raise ValueError(f"Must have 0 < low < high in: {self}")

    def sample(
        self,
        size: int,
        rng: np.random.Generator,
        normalscoresamples: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        if normalscoresamples is not None:
            return scipy.stats.loguniform.ppf(
                scipy.stats.norm.cdf(normalscoresamples), a=self.low, b=self.high
            )
        else:
            uniform_samples = generate_stratified_samples(size, rng)
            return scipy.stats.loguniform.ppf(uniform_samples, a=self.low, b=self.high)


@dataclasses.dataclass
class Triangular(Distribution):
    low: float = 0
    mode: float = 0.5
    high: float = 1.0

    def __post_init__(self) -> None:
        super().validate_params()

    def sample(
        self,
        size: int,
        rng: np.random.Generator,
        normalscoresamples: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        scale = self.high - self.low
        shape = (self.mode - self.low) / scale

        if normalscoresamples is not None:
            return scipy.stats.triang.ppf(
                scipy.stats.norm.cdf(normalscoresamples),
                shape,
                loc=self.low,
                scale=scale,
            )
        else:
            uniform_samples = generate_stratified_samples(size, rng)
            return scipy.stats.triang.ppf(
                uniform_samples, shape, loc=self.low, scale=scale
            )


@dataclasses.dataclass
class PERT(Distribution):
    """Create a PERT random variable, which is a re-parametrization of Beta.

    Example:
    >>> rng = np.random.default_rng(0)
    >>> PERT(low=0, high=10, mode=4).sample(5, rng=rng)
    array([6.11670474, 3.39187484, 1.45699256, 4.6543798 , 5.46365429])
    """

    low: float = 0
    mode: float = 0.5
    high: float = 1.0
    scale: float = 4.0

    def __post_init__(self) -> None:
        super().validate_params()

    def sample(
        self,
        size: int,
        rng: np.random.Generator,
        normalscoresamples: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        muval = (self.low + self.high + self.scale * self.mode) / (self.scale + 2)
        if np.isclose(muval, self.mode):
            alpha1 = (self.scale / 2) + 1
        else:
            alpha1 = ((muval - self.low) * (2 * self.mode - self.low - self.high)) / (
                (self.mode - muval) * (self.high - self.low)
            )
        alpha2 = alpha1 * (self.high - muval) / (muval - self.low)

        if normalscoresamples is not None:
            uniform_samples = scipy.stats.norm.cdf(normalscoresamples)
        else:
            uniform_samples = generate_stratified_samples(size, rng)

        values = scipy.stats.beta.ppf(uniform_samples, alpha1, alpha2)
        # Scale the beta distribution to the desired range
        return values * (self.high - self.low) + self.low


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
    return sampler.random(n=numreals).flatten()


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
        array with sampled values
    """
    if numreals < 0:
        raise ValueError("Got < 0 samples ({numreals=}) for distribution: {distname}")
    if numreals == 0:
        return np.array([])

    # Special case for discrete
    if distname.lower().startswith("disc"):
        return sample_discrete(dist_parameters, numreals, rng, normalscoresamples)

    # Special case for constant
    if distname.lower().startswith("const"):
        if normalscoresamples is not None:
            raise ValueError(
                "Parameter with const distribution was defined in correlation "
                "matrix but const distribution cannot be used with correlation."
            )
        return np.array([dist_parameters[0]] * numreals)

    # Convert parameters
    parameters: list[float] = validate_params(
        distname=distname, parameters=list(dist_parameters)
    )

    if distname.lower().startswith("norm"):
        if len(parameters) == 2:
            mean, stddev = parameters
            distr = Normal(mean=mean, stddev=stddev)
        elif len(parameters) == 4:
            mean, stddev, low, high = parameters
            distr = Normal(mean=mean, stddev=stddev, low=low, high=high)
        else:
            raise ValueError(f"Normal must have 2 or 4 parameters, got: {parameters}")
        return distr.sample(
            size=numreals, rng=rng, normalscoresamples=normalscoresamples
        )

    elif distname.lower().startswith("logn"):
        if len(parameters) != 2:
            raise ValueError(f"Lognormal must have 2 parameters, got: {parameters}")
        mean, sigma = parameters
        return Lognormal(mean=mean, sigma=sigma).sample(
            size=numreals, rng=rng, normalscoresamples=normalscoresamples
        )
    elif distname.lower().startswith("unif"):
        if len(parameters) != 2:
            raise ValueError(f"Uniform must have 2 parameters, got: {parameters}")
        low, high = parameters
        return Uniform(low=low, high=high).sample(
            size=numreals, rng=rng, normalscoresamples=normalscoresamples
        )
    elif distname.lower().startswith("triang"):
        if len(parameters) != 3:
            raise ValueError(f"Triangular must have 3 parameters, got: {parameters}")
        low, mode, high = parameters
        return Triangular(low=low, mode=mode, high=high).sample(
            size=numreals, rng=rng, normalscoresamples=normalscoresamples
        )
    elif distname.lower().startswith("pert"):
        if len(parameters) == 3:
            low, mode, high = parameters
            pert = PERT(low=low, mode=mode, high=high)
        elif len(parameters) == 4:
            low, mode, high, scale = parameters
            pert = PERT(low=low, mode=mode, high=high, scale=scale)
        else:
            raise ValueError(f"PERT must have 3 or 4 parameters, got: {parameters}")
        return pert.sample(
            size=numreals, rng=rng, normalscoresamples=normalscoresamples
        )
    elif distname.lower().startswith("logunif"):
        if len(parameters) != 2:
            raise ValueError(f"Loguniform must have 2 parameters, got: {parameters}")
        low, high = parameters
        return Loguniform(low=low, high=high).sample(
            size=numreals, rng=rng, normalscoresamples=normalscoresamples
        )
    else:
        raise ValueError(f"Distribution name {distname} is not implemented")


def sample_discrete(
    dist_params: Sequence[str],
    numreals: int,
    rng: np.random.Generator,
    normalscoresamples: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    """Sample discrete variables.

    Examples:
    >>> rng = np.random.default_rng(0)
    >>> sample_discrete(['a,b,c'], numreals=5, rng=rng)
    array(['c', 'b', 'a', 'b', 'c'], dtype='<U1')

    >>> sample_discrete(['a,b,c', '1,2,3'], numreals=5, rng=rng)
    array(['a', 'c', 'c', 'c', 'b'], dtype='<U1')

    >>> normalscoresamples = np.array([-0.5, 0.3, 0.4, 0.7, 0.7])
    >>> sample_discrete(['a,b,c', '1,2,3'], numreals=5, rng=rng,
    ...                 normalscoresamples=normalscoresamples)
    array(['b', 'c', 'c', 'c', 'c'], dtype='<U1')
    """
    outcomes = re.split(",", dist_params[0])
    outcomes = [item.strip() for item in outcomes]
    if numreals == 0:
        return np.array([])
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
    return values


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
