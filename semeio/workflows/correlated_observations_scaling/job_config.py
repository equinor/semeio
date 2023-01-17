from copy import deepcopy

import configsuite
from configsuite import MetaKeys as MK
from configsuite import types

from semeio.workflows.correlated_observations_scaling.exceptions import ValidationError
from semeio.workflows.correlated_observations_scaling.obs_utils import (
    find_and_expand_wildcards,
)
from semeio.workflows.correlated_observations_scaling.validator import (
    has_keys,
    is_subset,
)


@configsuite.validator_msg("Minimum length of index list must be > 1 for PCA")
def _min_length(value):
    return len(value) > 1


@configsuite.validator_msg("Minimum value of index must be >= 0")
def _min_value(value):
    return value >= 0


_NUM_CONVERT_MSG = "Will go through the input and try to convert to list of int"


@configsuite.transformation_msg(_NUM_CONVERT_MSG)
def _to_int_list(value):
    value = deepcopy(value)
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        value = ",".join([str(x) for x in value])
    return _realize_list(value)


@configsuite.transformation_msg("Convert ranges and singeltons into list")
def _realize_list(input_string):
    """If input_string is not a string, input_string will be returned. If input_string
    is a string it is assumed to contain comma separated elements. Each element is
    assumed to be either a singelton or a range. A singleton is a single number,
    i.e.  7 or 14. A range is a lower and upper element of the range separated by a
    single '-'. When provided with a string we will either return a list containing the
    union of all the singeltons and the ranges, or raise a TypeError or ValueError if
    it fails in the process. _realize_list('1,2,4-7,14-15') ->
    [1, 2, 4, 5, 6, 7, 14, 15]
    """
    real_list = []
    for elem in input_string.split(","):
        if elem.startswith("-"):
            raise ValueError(
                "Elements can not be negative, neither singletons nor in range,"
                f" got: {elem}"
            )
        bounds = elem.split("-")
        if len(bounds) == 1:
            real_list.append(int(elem))
        elif len(bounds) == 2:
            if elem.count("-") != 1:
                raise ValueError("Did expect single '-' in range")
            lower_bound = int(bounds[0])
            upper_bound = int(bounds[1]) + 1

            if lower_bound > upper_bound:
                err_msg = "Lower bound of range expected to be smaller then upper bound"
                raise ValueError(err_msg)

            real_list += range(lower_bound, upper_bound)
        else:
            raise ValueError("Expected at most one '-' in an element")

    return real_list


_NUM_CONVERT_MSG = "Create UPDATE_KEYS from CALCULATE_KEYS as it was not specified"


@configsuite.transformation_msg(_NUM_CONVERT_MSG)
def _expand_input(input_value):
    expanded_values = deepcopy(input_value)
    if "CALCULATE_KEYS" in expanded_values and "UPDATE_KEYS" not in expanded_values:
        expanded_values.update(
            {"UPDATE_KEYS": {"keys": expanded_values["CALCULATE_KEYS"]["keys"]}}
        )
    return expanded_values


@configsuite.validator_msg("Threshold must be higher than 0 and lower than 1")
def _min_max_value(value):
    return 0.0 < value < 1.0


@configsuite.validator_msg("keys must be provided for CALCULATE_KEYS")
def _CALCULATE_KEYS_key_not_empty_list(content):
    # pylint: disable=invalid-name
    return len(content) > 0


_CALCULATE_KEYS_DESCRIPTION = """
The keys that will be used for calculating a scaling factor and update all
data points within said keys. Include the indexes under "index" in order to
update selected data points. Use UPDATE_KEYS if you would like the scaling
to be applied to other keys than the ones listed under CALCULATE_KEYS.
The configuration accepts a list of CALCULATE_KEYS, facilitating update on
multiple groups for a single run, instead of having to run multiple times.

NB: Between runs on clusters, "threshold", "std_cutoff" and "std_cutoff" are
reset.

Example:

.. code-block:: yaml

    CALCULATE_KEYS:
        keys:
            -
                key: FOPR
             index: 1-10,50-100

This will calculate the scaling factor from indices 1-10 and 50-100, as
well as update these indices.
"""

_UPDATE_KEYS_DESCRIPTION = """
Unless provided, the keys to be updated are the same as the ones in
CALCULATE_KEYS. UPDATE_KEYS can be used to specify different indexes and or
keys to apply the scaling factor.

.. code-block:: yaml

    CALCULATE_KEYS:
        keys:
            -
                key: FOPR
                index: 1-10,50-100
    UPDATE_KEYS:
        keys:
            -
                key: FOPR
                index: 50-100

This configuration will calculate a scaling factor from indices 1-10,50-100
on "FOPR", but only update the scaling on indices "50-100".
"""

_KEYS_SCHEMA = {
    MK.ElementValidators: (_CALCULATE_KEYS_key_not_empty_list,),
    MK.Type: types.List,
    MK.Content: {
        MK.Item: {
            MK.Type: types.NamedDict,
            MK.Content: {
                "key": {
                    MK.Type: types.String,
                    MK.Description: "Name of the key. An asterisk is accepted "
                    "as a suffix to expand all matching keywords (e.g. "
                    "WOPR* will include WOPR:OP1)",
                },
                "index": {
                    MK.Type: types.List,
                    MK.LayerTransformation: _to_int_list,
                    MK.Description: "Indexes matching the data points relevant "
                    "for update. Accepts single integer and ranges e.g. "
                    "(1,2,4-6,14-15) ->[1, 2, 4, 5, 6, 14, 15]",
                    MK.Content: {
                        MK.Item: {
                            MK.Type: types.Integer,
                            MK.ElementValidators: (_min_value,),
                        }
                    },
                },
            },
        }
    },
}


_CORRELATED_OBSERVATIONS_SCHEMA = {
    MK.Type: types.NamedDict,
    MK.Description: "Keys and index lists from all scaled keys",
    MK.LayerTransformation: _expand_input,
    MK.Content: {
        "CALCULATE_KEYS": {
            MK.Type: types.NamedDict,
            MK.Description: _CALCULATE_KEYS_DESCRIPTION,
            MK.Content: {
                "keys": _KEYS_SCHEMA,
                "threshold": {
                    MK.Type: types.Number,
                    MK.ElementValidators: (_min_max_value,),
                    MK.Description: "Threshold used when computing primary"
                    "components of the clusters.",
                    MK.Default: 0.95,
                },
                "std_cutoff": {
                    MK.Type: types.Number,
                    MK.Description: "A lower bound on the ensemble standard"
                    " deviation. All data points with insufficient variation"
                    " will be dropped. The value is defaulted to the value used"
                    " in the ert run, but if a value is given in the config, "
                    " that value will be used",
                },
                "alpha": {
                    MK.Type: types.Number,
                    MK.Description: "Scalar controlling the allowed distance"
                    "between ensemble mean and observation. In particular, "
                    "if: `abs(observed_value - ensemble_mean) > "
                    "alpha * (ensenmble_std + observed_std)` the data point "
                    "will be dropped. The value is defaulted to the value used "
                    "in the ert run, but if a value is given in the config, "
                    "that value will be used",
                },
            },
        },
        "UPDATE_KEYS": {
            MK.Type: types.NamedDict,
            MK.Description: _UPDATE_KEYS_DESCRIPTION,
            MK.Content: {"keys": _KEYS_SCHEMA},
        },
    },
}


class ObsCorrConfig:
    def __init__(self, config_dict, obs_keys, default_values):
        self._config_dict = config_dict
        self._obs_keys = obs_keys
        self._default_values = default_values
        self.config = self._setup_configuration()
        self.errors = []

    def validate(self):
        """
        Validates the full configuration. If invalid, raises an ValidationError.
        """
        self._schema_validation()
        self._context_validation()

        if len(self.errors) > 0:
            raise ValidationError("Invalid job", self.errors)

    def _schema_validation(self):
        self.errors.extend(list(self.config.errors))

    def _context_validation(self):
        config = self.config.snapshot
        calc_keys = [event.key for event in config.CALCULATE_KEYS.keys]
        application_keys = [entry.key for entry in config.UPDATE_KEYS.keys]
        self.errors.extend(is_subset(calc_keys, application_keys))
        self.errors.extend(
            has_keys(self._obs_keys, calc_keys, "Key: {} has no observations")
        )

    def _setup_configuration(self):
        """
        Creates a ConfigSuite instance and inserts default values
        """
        config_data = self._config_dict
        obs_keys = self._obs_keys
        default_values = self._default_values
        config_dict = find_and_expand_wildcards(obs_keys, config_data)
        config = configsuite.ConfigSuite(
            config_dict,
            _CORRELATED_OBSERVATIONS_SCHEMA,
            deduce_required=True,
            layers=(default_values,),
        )
        return config

    def get_calculation_keys(self):
        return [event.key for event in self.config.snapshot.CALCULATE_KEYS.keys]

    def get_index_lists(self):
        return [
            event.index if len(event.index) > 0 else None
            for event in self.config.snapshot.CALCULATE_KEYS.keys
        ]

    def get_update_keys(self):
        return self.config.snapshot.UPDATE_KEYS.keys

    def get_threshold(self):
        return self.config.snapshot.CALCULATE_KEYS.threshold

    def get_alpha(self):
        return self.config.snapshot.CALCULATE_KEYS.alpha

    def get_std_cutoff(self):
        return self.config.snapshot.CALCULATE_KEYS.std_cutoff
