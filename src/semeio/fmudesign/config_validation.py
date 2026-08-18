"""
Module for validation of config (typically read from Excel).
"""

import copy
from enum import StrEnum
from typing import Any


class ConfigValidationError(ValueError):
    pass


class ConversionError(ValueError):
    def __init__(self, val: Any, key: str, type_str: str) -> None:  # ruff: ignore[any-type]
        raise ConfigValidationError(
            f"Could not convert '{val}' to {type_str} for key '{key}'. "
            f"Failed to validate '{val}'",
        )


def _validate_key_in_config(
    key: str, config: dict[str, Any], err_msg: str | None = None
) -> None:
    if key not in config:
        if err_msg is None:
            err_msg = f"'{key}' must be specified in general input sheet"
        raise ConfigValidationError(err_msg)


def _validate_int(maybe_int: Any, key: str) -> int:  # ruff: ignore[any-type]
    try:
        return int(maybe_int)
    except (TypeError, ValueError) as e:
        raise ConversionError(maybe_int, key, "int") from e


def _validate_positive_int(maybe_int: Any, key: str) -> int:  # ruff: ignore[any-type]
    try:
        int_ = int(maybe_int)
    except Exception as e:
        raise ConversionError(maybe_int, key, "int") from e

    if int_ < 0:
        raise ConfigValidationError(
            f"'{int_}' must be a positive integer for key '{key}'. "
            f"Failed to validate '{int_}'",
        )
    return int_


def _validate_none(maybe_none: Any, key: str) -> None:  # ruff: ignore[any-type]
    is_nonetype = maybe_none is None
    is_nonestring = False
    if isinstance(maybe_none, str):
        is_nonestring = maybe_none.lower() in {"", "none", "null"}
    if not is_nonetype or not is_nonestring:
        raise ConversionError(maybe_none, key, "None")


def _validate_string(maybe_string: Any, key: str) -> str:  # ruff: ignore[any-type]
    try:
        return str(maybe_string)
    except Exception as e:
        raise ConversionError(maybe_string, key, "str") from e


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


def _validate_no_extra_keys(config) -> None:
    ALLOWED_KEYS = {
        "designtype",
        "repeats",
        "correlation_iterations",
        "distribution_seed",
        "seed_strategy",
        "rms_seeds",
        "background",
    }
    extra_keys = set(config.keys()) - set(ALLOWED_KEYS)
    if extra_keys:
        msg = (
            "In the general input sheet, the following parameter(s) are not"
            f"recognized and cannot be parsed:\n{extra_keys!r}\n"
            f"Allowed keys:{ALLOWED_KEYS!r}"
        )
        raise LookupError(msg)


def _validate_designtype(config: dict[str, Any]) -> None:
    _validate_key_in_config("designtype", config)
    if config["designtype"] != "onebyone":
        raise ConfigValidationError(
            "Generation of DesignMatrix only implemented for designtype 'onebyone', "
            f"not '{config['designtype']}'"
        )


def _validate_repeats(config: dict[str, Any]) -> None:
    _validate_key_in_config("repeats", config)
    _validate_int(config["repeats"], "repeats")


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
        config[key] = _validate_positive_int(config[key], "correlation_iterations")


def _validate_distribution_seed(config: dict[str, Any]) -> None:
    key = "distribution_seed"
    _validate_key_in_config(
        key,
        config,
        err_msg=(
            "You did not specify a value for 'distribution_seed', which is used to "
            "seed the random number generator that draws from distributions in Monte "
            "Carlo sensitivities.\n"
            "- Specify a number (e.g. a 6 digit integer) to seed the random number "
            "generator and obtain reproducible results.\n"
            "- Specify None if you do not want to seed the random number generator. "
            "Your analysis will not be reproducible."
        ),
    )
    dist_seed = config["distribution_seed"]
    try:
        _validate_positive_int(dist_seed, "distribution_seed")
    except ConfigValidationError as e:
        if dist_seed is not None:
            raise ConfigValidationError(
                f"'{dist_seed}' must be a positive integer or None for key '{key}'. "
                f"Failed to validate '{dist_seed}'",
            ) from e


def _validate_seed_strategy(config: dict[str, Any]) -> None:
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


def _validate_rms_seeds(config: dict[str, Any]) -> None:
    key = "rms_seeds"
    try:
        value = _validate_string(config.get(key), key)
    except ConversionError:
        value = _validate_none(config.get(key), key)
    except ConfigValidationError:
        raise ConfigValidationError("")


def validate_general_input(
    config: dict[str, Any], verbosity: int = 0
) -> dict[str, Any]:
    """Main function for config validation.

    This function is responsible for:
        - Checking that required keys exist
        - Checking that values are set to valid types
        - Setting default values if keys are not set

    """
    config = copy.deepcopy(config)

    _validate_no_extra_keys(config)
    _validate_designtype(config)
    _validate_repeats(config)
    _validate_correlation_iterations(config, verbosity)
    _validate_distribution_seed(config)
    _validate_rms_seeds(config)

    return config
