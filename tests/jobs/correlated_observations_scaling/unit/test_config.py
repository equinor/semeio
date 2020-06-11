from copy import deepcopy

import pytest
import configsuite

from semeio.jobs.correlated_observations_scaling.job_config import (
    _expand_input,
    _min_value,
    _to_int_list,
    build_schema,
)


@pytest.mark.parametrize(
    "valid_input",
    [
        "0,1,2-5",
        [0, 1, "2-5"],
        [0, 1, 2, 3, 4, 5],
        [0, 1, "2-3", "4-5"],
        "0-5",
        "0-1,2,3-5",
        ["0,1,2-5"],
    ],
)
def test_to_int_list(valid_input):
    expected_result = list(range(6))
    assert _to_int_list(valid_input) == expected_result


@pytest.mark.parametrize(
    "test_input,expected_result", [(-1, False), (0, True), (1, True)]
)
def test_min_value(test_input, expected_result):
    assert _min_value(test_input).__bool__() == expected_result


def test_expand_input_no_modification():
    expected_result = {
        "UPDATE_KEYS": {"keys": [{"key": "key_4"}, {"key": "key_5"}, {"key": "key_6"}]},
        "CALCULATE_KEYS": {
            "keys": [{"key": "key_1"}, {"key": "key_2"}, {"key": "key_3"}],
            "threshold": 1.0,
        },
    }

    assert _expand_input(deepcopy(expected_result)) == expected_result


def test_expand_input_modification():
    keys = [{"key": "key_1"}, {"key": "key_2"}, {"key": "key_3"}]
    test_input = {"CALCULATE_KEYS": {"keys": keys, "threshold": 1.0}}

    expected_results = deepcopy(test_input)
    expected_results["UPDATE_KEYS"] = {"keys": keys}

    assert _expand_input(test_input) == expected_results


def test_valid_config_setup():

    valid_config_data = {
        "CALCULATE_KEYS": {"keys": [{"key": "first_key"}, {"key": "second_key"}]}
    }

    schema = build_schema()
    config = configsuite.ConfigSuite(valid_config_data, schema, deduce_required=True,)
    assert config.valid

    valid_config_data = {
        "CALCULATE_KEYS": {"keys": [{"key": "first_key"}, {"key": "second_key"}]},
        "UPDATE_KEYS": {"keys": [{"index": [1, 2, 3], "key": "first_key"}]},
    }

    schema = build_schema()
    config = configsuite.ConfigSuite(valid_config_data, schema, deduce_required=True,)
    assert config.valid


@pytest.mark.parametrize(
    "test_input,expected_errors",
    [
        pytest.param(
            {"UPDATE_KEYS": {"keys": [{"index": "1", "key": "a_key"}]}},
            ["keys must be provided for CALCULATE_KEYS is false on input '()'"],
            id="invalid_missing_required_CALCULATE_KEYS",
        ),
        pytest.param(
            {"CALCULATE_KEYS": {"keys": [{"index": "1"}]}},
            # The error msg is applied to both CALCULATE_KEYS and UPDATE_KEYS
            ["Missing key: key", "Missing key: key"],
            id="invalid_missing_required_CALCULATE_KEYS_keys",
        ),
        pytest.param(
            {
                "CALCULATE_KEYS": {
                    "keys": [{"key": "first_key"}, {"key": "second_key"}]
                },
                "UPDATE_KEYS": {"keys": [{"index": [-1, 2, 3], "key": "first_key"}]},
            },
            [
                "Elements can not be negative",
                "Minimum value of index must be >= 0 is false on input '-1'",
            ],
            id="invalid_negative_index",
        ),
    ],
)
def test_invalid_config_setup(test_input, expected_errors):
    schema = build_schema()
    config = configsuite.ConfigSuite(test_input, schema, deduce_required=True,)
    assert not config.valid

    msgs = [e.msg for e in config.errors]
    assert len(expected_errors) == len(msgs)
    assert all([any(exp in msg for msg in msgs) for exp in expected_errors])


def test_valid_configuration():
    valid_config_data = {
        "CALCULATE_KEYS": {"keys": [{"key": "POLY_OBS"}]},
        "UPDATE_KEYS": {"keys": [{"key": "POLY_OBS"}]},
    }

    schema = build_schema()
    config = configsuite.ConfigSuite(valid_config_data, schema, deduce_required=True,)

    assert config.valid
