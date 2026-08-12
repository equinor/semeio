"""
Module for validation of config (typically read from Excel).
"""

import copy
import numbers
from enum import StrEnum
from typing import Any


class ConfigValidationError(ValueError):
    pass


class SeedStrategy(StrEnum):
    """How Monte Carlo samples are seeded.

    JOINT:
        All parameters are drawn in one Latin Hypercube Sampling call (the
        default). Adding, removing or reordering a parameter reshuffles every
        other parameter.
    INDEPENDENT:
        Each parameter, and each correlation group, is seeded separately from
        the base seed, so changing one leaves the others bit-identical.
    """

    JOINT = "joint"
    INDEPENDENT = "independent"


def _validate_designtype(config: dict[str, Any]) -> None:
    if "designtype" not in config:
        raise ConfigValidationError(
            "'designtype' must be specified in general input sheet"
        )
    if config["designtype"] != "onebyone":
        raise ConfigValidationError(
            "Generation of DesignMatrix only implemented for type 'onebyone', "
            f"not '{config['designtype']}'"
        )


def _validate_repeats(config: dict[str, Any]) -> None:
    if "repeats" not in config:
        raise ConfigValidationError(
            "'repeats' must be specified in general input sheet"
        )
    if not isinstance(config["repeats"], int):
        raise ConfigValidationError(
            f"'repeats' in general_input must be an int, "
            f"got 'repeats = {config['repeats']}' "
            f"with type: {type(config['repeats']).__name__}"
        )


def _validate_correlation_iterations(config: dict[str, Any], verbosity: int) -> None:
    key = "correlation_iterations"
    if key not in config:
        if verbosity > 0:
            print(f"{key!r} not set in general input sheet. Setting to default 0.")
            print("  - When set to 0, Iman Conover is used to induce correlations.")
            print(
                "  - When set to a positive integer N, Iman Conover is followed by N iterations\n"  # ruff: ignore[line-too-long]
                "    of random permutations (swaps). This leads to results that are never worse, and often better.\n"  # ruff: ignore[line-too-long]
                "    It is especially useful for skewed distributions like lognormal and high dimensional problems."  # ruff: ignore[line-too-long]
            )
            print(
                f"  If desired correlation does not match observed, try setting {key!r}=999 or higher."  # ruff: ignore[line-too-long]
            )
        config[key] = 0
    else:
        try:
            config[key] = int(config[key])
        except (ValueError, TypeError) as err:
            raise ConfigValidationError(
                f"{key!r} must be a non-negative integer, got: {config[key]}"
            ) from err


def _validate_distribution_seed(config: dict[str, Any]) -> None:
    if "distribution_seed" not in config:
        raise ConfigValidationError(
            "You did not specify a value for 'distribution_seed', which is used to "
            "seed the random number generator that draws from distributions in Monte "
            "Carlo sensitivities.\n"
            "- Specify a number (e.g. a 6 digit integer) to seed the random number "
            "generator and obtain reproducible results.\n"
            "- Specify None if you do not want to seed the random number generator. "
            "Your analysis will not be reproducible."
        )

    key = "distribution_seed"
    if key not in config:
        raise ConfigValidationError
    if not (isinstance(config[key], numbers.Integral) or (config[key] is None)):
        raise ConfigValidationError(
            f"{key!r} must be a non-negative integer or None, got: {config[key]}"
        )


def _validate_rms_seeds(config: dict[str, Any]) -> None:
    # 'seed_strategy' controls how Monte Carlo samples are seeded.
    # See the SeedStrategy docstring for what each strategy means.
    key = "seed_strategy"
    value = config.get(key)
    if isinstance(value, str):
        value = value.strip().lower()
    if value is None or value == "none":
        value = SeedStrategy.JOINT
    try:
        config[key] = SeedStrategy(value)
    except (ValueError, TypeError) as err:
        raise ConfigValidationError(
            f"{key!r} must be one of {[s.value for s in SeedStrategy]}, "
            f"got: {config[key]}"
        ) from err


def validate_configuration(
    config: dict[str, Any], verbosity: int = 0
) -> dict[str, Any]:
    """Main function for config validation.

    This function is responsible for:
        - Checking that required keys exist
        - Checking that values are set to valid types
        - Setting default values if keys are not set

    """
    config = copy.deepcopy(config)

    _validate_designtype(config)
    _validate_repeats(config)
    _validate_correlation_iterations(config, verbosity)
    _validate_distribution_seed(config)
    _validate_rms_seeds(config)

    return config
