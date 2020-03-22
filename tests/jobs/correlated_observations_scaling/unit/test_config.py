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


def test_config_setup():

    valid_config_data = {
        "CALCULATE_KEYS": {"keys": [{"key": "first_key"}, {"key": "second_key"}]}
    }

    schema = build_schema()
    config = configsuite.ConfigSuite(valid_config_data, schema)
    assert config.valid

    valid_config_data = {
        "CALCULATE_KEYS": {"keys": [{"key": "first_key"}, {"key": "second_key"}]},
        "UPDATE_KEYS": {"keys": [{"index": [1, 2, 3], "key": "first_key"}]},
    }

    schema = build_schema()
    config = configsuite.ConfigSuite(valid_config_data, schema)
    assert config.valid

    invalid_too_short_index_list = {
        "UPDATE_KEYS": {"keys": [{"index": "1", "key": ["a_key"]}]}
    }

    config = configsuite.ConfigSuite(invalid_too_short_index_list, schema)
    assert not config.valid

    invalid_missing_required_keyword = {
        "CALCULATE_KEYS": {"keys": [{"key": "a_key"}]},
        "UPDATE_KEYS": {"index": "1-5"},
    }

    config = configsuite.ConfigSuite(invalid_missing_required_keyword, schema)
    assert not config.valid

    invalid_negative_index = {
        "CALCULATE_KEYS": {"keys": [{"key": "first_key"}, {"key": "second_key"}]},
        "UPDATE_KEYS": {"keys": [{"index": [-1, 2, 3], "key": "first_key"}]},
    }

    schema = build_schema()
    config = configsuite.ConfigSuite(invalid_negative_index, schema)
    assert not config.valid


def test_valid_configuration():
    valid_config_data = {
        "CALCULATE_KEYS": {"keys": [{"key": "POLY_OBS"}]},
        "UPDATE_KEYS": {"keys": [{"key": "POLY_OBS"}]},
    }

    schema = build_schema()
    config = configsuite.ConfigSuite(valid_config_data, schema)

    assert config.valid
