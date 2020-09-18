import os
import re
import shutil

import configsuite
import pytest
from res.enkf import EnKFMain, ResConfig
from semeio.workflows.correlated_observations_scaling import job_config
from semeio.workflows.correlated_observations_scaling.exceptions import ValidationError
from semeio.workflows.correlated_observations_scaling.obs_utils import (
    _wildcard_to_dict_list,
    create_active_lists,
    find_and_expand_wildcards,
    keys_with_data,
)


@pytest.mark.parametrize(
    "matching_keys,entry,expected_result",
    [
        (["a_key"], {"key": "a_*"}, [{"key": "a_key"}]),
        (["a_key", "b_key"], {"key": "*key"}, [{"key": "a_key"}, {"key": "b_key"}]),
        (
            ["a_key"],
            {"key": "a_*", "index": [1, 2]},
            [{"key": "a_key", "index": [1, 2]}],
        ),
    ],
)
def test_wildcard_to_dict_list(matching_keys, entry, expected_result):
    assert _wildcard_to_dict_list(matching_keys, entry) == expected_result


def test_find_and_expand_wildcards():
    expected_dict = {
        "ANOTHER_KEY": "something",
        "CALCULATE_KEYS": {
            "keys": [
                {"key": "WOPR_OP1_108"},
                {"key": "WOPR_OP1_144"},
                {"key": "WOPR_OP1_190"},
                {"key": "WOPR_OP1_9"},
                {"key": "WOPR_OP1_36"},
                {"key": "WOPR_OP1_72"},
                {"key": "FOPR"},
            ]
        },
        "UPDATE_KEYS": {
            "keys": [
                {"key": "WOPR_OP1_108"},
                {"key": "WOPR_OP1_144"},
                {"key": "WOPR_OP1_190"},
                {"key": "FOPR"},
            ]
        },
    }

    user_config = {
        "ANOTHER_KEY": "something",
        "CALCULATE_KEYS": {"keys": [{"key": "WOPR_*"}, {"key": "FOPR"}]},
        "UPDATE_KEYS": {"keys": [{"key": "WOPR_OP1_1*"}, {"key": "FOPR"}]},
    }

    observation_list = [
        "WOPR_OP1_108",
        "WOPR_OP1_144",
        "WOPR_OP1_190",
        "WOPR_OP1_9",
        "WOPR_OP1_36",
        "WOPR_OP1_72",
        "FOPR",
    ]

    result_dict = find_and_expand_wildcards(observation_list, user_config)

    assert result_dict == expected_dict


@pytest.mark.parametrize(
    "config_dict,obs_list,expected_fails,err_msg",
    [
        (
            {"CALCULATE_KEYS": {"keys": [{"key": "FOP*"}]}},
            ["FOPR"],
            False,
            None,
        ),
        (
            {"CALCULATE_KEYS": {"keys": [{"key": "TYPO*"}, {"key": "FOP*"}]}},
            ["FOPR"],
            True,
            "Invalid CALCULATE_KEYS TYPO* had no match",
        ),
        (
            {"CALCULATE_KEYS": {"keys": [{"key": "TYPO*"}]}},
            ["FOPR"],
            True,
            "Invalid CALCULATE_KEYS TYPO* had no match",
        ),
        (
            {
                "CALCULATE_KEYS": {"keys": [{"key": "FOPR*"}]},
                "UPDATE_KEYS": {"keys": [{"key": "TYPO*"}]},
            },
            ["FOPR"],
            True,
            "Invalid UPDATE_KEYS TYPO* had no match",
        ),
        (
            {"CALCULATE_KEYS": {"keys": [{"key": "TYPO*"}, {"key": "FOPR"}]}},
            ["FOPR"],
            True,
            "Invalid CALCULATE_KEYS TYPO* had no match",
        ),
    ],
)
def test_failed_wildcard_expansion(config_dict, obs_list, expected_fails, err_msg):
    if expected_fails:
        with pytest.raises(ValidationError) as excinfo:
            find_and_expand_wildcards(obs_list, config_dict)
        # ignore tabs, newlines
        exc_msg = re.sub("\\s+", " ", str(excinfo.value))
        assert exc_msg == err_msg
    else:
        try:
            find_and_expand_wildcards(obs_list, config_dict)
        except ValueError:
            pytest.fail("unexpectedly raised with config: {}".format(config_dict))


@pytest.mark.usefixtures("setup_ert")
def test_create_observation_vectors(setup_ert):

    valid_config_data = {
        "CALCULATE_KEYS": {"keys": [{"key": "WPR_DIFF_1"}]},
        "UPDATE_KEYS": {"keys": [{"key": "WPR_DIFF_1"}]},
    }
    config = configsuite.ConfigSuite(
        valid_config_data,
        job_config._CORRELATED_OBSERVATIONS_SCHEMA,
        deduce_required=True,
    )

    res_config = setup_ert
    ert = EnKFMain(res_config)
    obs = ert.getObservations()

    new_events = create_active_lists(obs, config.snapshot.UPDATE_KEYS.keys)

    keys = [event.key for event in new_events]

    assert "WPR_DIFF_1" in keys
    assert "SNAKE_OIL_WPR_DIFF" not in keys


@pytest.mark.usefixtures("setup_tmpdir")
def test_add_observation_vectors(test_data_root):

    valid_config_data = {"UPDATE_KEYS": {"keys": [{"key": "WOPR_OP1_108"}]}}

    schema = job_config._CORRELATED_OBSERVATIONS_SCHEMA
    config = configsuite.ConfigSuite(valid_config_data, schema, deduce_required=True)

    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")

    ert = EnKFMain(res_config)

    obs = ert.getObservations()

    new_events = create_active_lists(obs, config.snapshot.UPDATE_KEYS.keys)

    keys = [event.key for event in new_events]

    assert "WOPR_OP1_108" in keys
    assert "WOPR_OP1_144" not in keys


@pytest.mark.usefixtures("setup_tmpdir")
def test_validate_failed_realizations(test_data_root):
    """
    Config has several failed realisations
    """
    test_data_dir = os.path.join(test_data_root, "failed_runs_in_storage")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("mini_fail_config")
    ert = EnKFMain(res_config)
    observations = ert.getObservations()

    result = keys_with_data(
        observations,
        ["GEN_PERLIN_1"],
        ert.getEnsembleSize(),
        ert.getEnkfFsManager().getCurrentFileSystem(),
    )
    assert result == ["GEN_PERLIN_1"]


@pytest.mark.usefixtures("setup_tmpdir")
def test_validate_no_realizations(test_data_root):
    """
    Ensamble has not run
    """
    test_data_dir = os.path.join(test_data_root, "poly_normal")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    observations = ert.getObservations()

    result = keys_with_data(
        observations,
        ["POLY_OBS"],
        ert.getEnsembleSize(),
        ert.getEnkfFsManager().getCurrentFileSystem(),
    )
    assert result == []
