# -*- coding: utf-8 -*-
import fnmatch

import configsuite

from copy import deepcopy
from collections import namedtuple

from semeio.jobs.correlated_observations_scaling import job_config
from semeio.jobs.correlated_observations_scaling.scaled_matrix import DataMatrix
from ert_data.measured import MeasuredData
from semeio.jobs.correlated_observations_scaling.validator import (
    valid_configuration,
    valid_job,
)
from res.enkf import LocalObsdata, ActiveList

from ecl.util.util import BoolVector
from res.enkf import RealizationStateEnum


def scaling_job(facade, user_config_dict):
    """
    Takes an instance of EnkFMain and a user config dict, will do some pre-processing on
    the user config dict, set up a ConfigSuite instance and validate the job before control
    is passed to the main job.
    """

    observation_keys = [
        facade.get_observation_key(nr) for nr, _ in enumerate(facade.get_observations())
    ]
    obs_with_data = keys_with_data(
        facade.get_observations(),
        observation_keys,
        facade.get_ensemble_size(),
        facade.get_current_fs(),
    )
    config_dict = _find_and_expand_wildcards(observation_keys, user_config_dict)

    config = setup_configuration(config_dict, job_config.build_schema())

    if not valid_configuration(config):
        raise ValueError("Invalid configuration")
    if not valid_job(config, observation_keys, obs_with_data):
        raise ValueError("Invalid job")
    _observation_scaling(facade, config.snapshot)


def _observation_scaling(facade, config):
    """
    Collects data, performs scaling and applies scaling, assumes validated input.
    """
    calculate_keys = [event.key for event in config.CALCULATE_KEYS.keys]
    index_lists = [event.index for event in config.CALCULATE_KEYS.keys]
    measured_data = MeasuredData(facade, calculate_keys, index_lists)
    measured_data.remove_failed_realizations()
    measured_data.remove_inactive_observations()
    measured_data.filter_ensemble_mean_obs(config.CALCULATE_KEYS.alpha)
    measured_data.filter_ensemble_std(config.CALCULATE_KEYS.std_cutoff)

    matrix = DataMatrix(measured_data.data)
    matrix.std_normalization(inplace=True)

    scale_factor = matrix.get_scaling_factor(config.CALCULATE_KEYS)

    update_data = _create_active_lists(
        facade.get_observations(), config.UPDATE_KEYS.keys
    )

    _update_scaling(facade.get_observations(), scale_factor, update_data)


def _wildcard_to_dict_list(matching_keys, entry):
    """
    One of either:
    entry = {"key": "WOPT*", "index": [1,2,3]}
    entry = {"key": "WOPT*"}
    """
    if "index" in entry:
        return [{"key": key, "index": entry["index"]} for key in matching_keys]
    else:
        return [{"key": key} for key in matching_keys]


def _expand_wildcard(obs_list, wildcard_key, entry):
    """
    Expands a wildcard
    """
    matching_keys = fnmatch.filter(obs_list, wildcard_key["key"])
    return _wildcard_to_dict_list(matching_keys, entry)


def _find_and_expand_wildcards(obs_list, user_dict):
    """
    Loops through the user input and identifies wildcards in observation
    names and expands them.
    """
    new_dict = deepcopy(user_dict)
    for main_key, value in user_dict.items():
        new_entries = []
        if main_key in ("UPDATE_KEYS", "CALCULATE_KEYS"):
            for val in value["keys"]:
                if "*" in val["key"]:
                    new_entries.extend(_expand_wildcard(obs_list, val, val["key"]))
                else:
                    new_entries.append(val)
            new_dict[main_key]["keys"] = new_entries

    return new_dict


def setup_configuration(input_data, schema):
    """
    Creates a ConfigSuite instance and inserts default values
    """
    default_layer = job_config.get_default_values()
    config = configsuite.ConfigSuite(input_data, schema, layers=(default_layer,))
    return config


def _create_active_lists(enkf_observations, events):
    """
    Will add observation vectors to observation data. Returns
    a list of tuples mirroring the user config but also containing
    the active list where the scaling factor will be applied.
    """
    new_events = []
    observation_data = LocalObsdata("some_name", enkf_observations)
    for event in events:
        observation_data.addObsVector(enkf_observations[event.key])

        obs_index = _data_index_to_obs_index(enkf_observations, event.key, event.index)
        new_active_list = _get_active_list(observation_data, event.key, obs_index)

        new_events.append(_make_tuple(event.key, event.index, new_active_list))

    return new_events


def _get_active_list(observation_data, key, index_list):
    """
    If the user doesn't supply an index list, the existing active
    list from the observation is used, otherwise an active list is
    created from the index list.
    """
    if index_list is not None:
        return _active_list_from_index_list(index_list)
    else:
        return observation_data.copy_active_list(key).setParent()


def _make_tuple(
    key,
    index,
    active_list,
    new_event=namedtuple("named_dict", ["key", "index", "active_list"]),
):
    return new_event(key, index, active_list)


def _update_scaling(obs, scale_factor, events):
    """
    Applies the scaling factor to the user specified index, SUMMARY_OBS needs to be treated differently
    as it only has one data point per node, compared with other observation types which have multiple
    data points per node.
    """
    for event in events:
        obs_vector = obs[event.key]
        for index, obs_node in enumerate(obs_vector):
            if obs_vector.getImplementationType().name == "SUMMARY_OBS":
                index_list = (
                    event.index if event.index is not None else range(len(obs_vector))
                )
                if index in index_list:
                    obs_node.set_std_scaling(scale_factor)
            elif obs_vector.getImplementationType().name != "SUMMARY_OBS":
                obs_node.updateStdScaling(scale_factor, event.active_list)
    print(
        "Keys: {} scaled with scaling factor: {}".format(
            [event.key for event in events], scale_factor
        )
    )


def _active_list_from_index_list(index_list):
    """
    Creates an ActiveList from a list of indexes
    :param index_list: list of index
    :type index_list:  list
    :return: Active list, a c-object with mode (ALL-ACTIVE, PARTIALLY-ACTIVE, INACTIVE) and list of indices
    :rtype: active_list
    """
    active_list = ActiveList()
    [active_list.addActiveIndex(index) for index in index_list]
    return active_list


def _set_active_lists(observation_data, key_list, active_lists):
    """
    Will make a backup of the existing active list on the observation node
    before setting the user supplied index list.
    """
    exisiting_active_lists = []
    for key, active_list in zip(key_list, active_lists):
        exisiting_active_lists.append(observation_data.copy_active_list(key))
        observation_data.setActiveList(key, active_list)
    return observation_data, exisiting_active_lists


def keys_with_data(observations, keys, ensamble_size, storage):
    """
    Checks that all keys have data and returns a list of error messages
    """
    active_realizations = storage.realizationList(RealizationStateEnum.STATE_HAS_DATA)

    if len(active_realizations) == 0:
        return []

    active_mask = BoolVector.createFromList(ensamble_size, active_realizations)
    return [key for key in keys if observations[key].hasData(active_mask, storage)]


def _data_index_to_obs_index(obs, obs_key, data_index_list):
    if obs[obs_key].getImplementationType().name != "GEN_OBS":
        return data_index_list
    elif data_index_list is None:
        return data_index_list

    for timestep in obs[obs_key].getStepList().asList():
        node = obs[obs_key].getNode(timestep)
        index_map = {node.getIndex(nr): nr for nr in range(len(node))}
    return [index_map[index] for index in data_index_list]
