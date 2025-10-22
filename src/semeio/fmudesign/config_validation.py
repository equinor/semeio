"""
Module for validation of config (typically read from Excel).
"""

import copy
import numbers

from semeio.fmudesign._excel_to_dict import resolve_path
from semeio.fmudesign.utils import seeds_from_extern


def validate_configuration(config: dict, verbosity: int = 0) -> dict:
    """Main function for config validation.

    This function is responsible for:
        - Checking that required keys exist
        - Checking that values are set to valid types
        - Setting default values if keys are not set

    """
    config = copy.deepcopy(config)

    if "repeats" not in config:
        raise LookupError('"repeats" must be specified in general input sheet')

    key = "correlation_iterations"
    if key not in config:
        if verbosity > 0:
            print(f"{key!r} not set in general input sheet. Setting to default 0.")
            print("  - When set to 0 Iman Conover is used to induce correlations.")
            print(
                "  - When set to a posotive integer, Iman Conover is followed by a heuristic algorithm."
            )
            print(
                f"  If desired correlation does not match observed, try setting {key!r}=999 or higher."
            )
        config[key] = 0
    else:
        try:
            config[key] = int(config[key])
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"{key!r} must be a non-negative integer, got: {config[key]}"
            ) from err

    key = "distribution_seed"
    if key not in config:
        raise ValueError(
            "You did not specify a value for 'distribution_seed', which is used to seed "
            "the random number generator that draws from distributions in Monte Carlo "
            "sensitivities.\n"
            "- Specify a number (e.g. a 6 digit integer) to seed the random number "
            "generator and obtain reproducible results.\n"
            "- Specify None if you do not want to seed the random number generator. "
            "Your analysis will not be reproducible."
        )
    if not (isinstance(config[key], numbers.Integral) or (config[key] is None)):
        raise ValueError(
            f"{key!r} must be a non-negative integer or None, got: {config[key]}"
        )

    # 'seeds' here is 'rms_seeds' in the input. It can be either:
    # - 'default' => gives seed numbers 1000, 1001, 1002, ...
    # - 'None'    => seed number not added
    # - a path to a file
    key = "seeds"
    if key not in config:
        msg = '"rms_seeds" must be specified in general input sheet\n'
        msg += ' - Set to "None", "default" or path to a file.'
        raise LookupError(msg)
    if not ((config[key] == "default") or (config[key] is None)):
        # It must be a path to a file with seed values
        try:
            path = resolve_path(config.get("input_file"), config[key])
            config[key] = seeds_from_extern(path)
        except Exception as exception:
            msg = f"'rms_seeds' must be 'None', 'default' or a file, got: {config[key]}"
            raise ValueError(msg) from exception

    return config
