import fnmatch
from collections import namedtuple
from copy import deepcopy

from ecl.util.util import BoolVector
from res.enkf import ActiveList, RealizationStateEnum
from semeio.workflows.correlated_observations_scaling.exceptions import ValidationError


def _wildcard_to_dict_list(matching_keys, entry):
    """
    One of either:
    entry = {"key": "WOPT*", "index": [1,2,3]}
    entry = {"key": "WOPT*"}
    """
    if "index" in entry:
        return [{"key": key, "index": entry["index"]} for key in matching_keys]
    return [{"key": key} for key in matching_keys]


def _expand_wildcard(obs_list, wildcard_key, entry):
    """
    Expands a wildcard
    """
    matching_keys = fnmatch.filter(obs_list, wildcard_key["key"])
    return _wildcard_to_dict_list(matching_keys, entry)


def find_and_expand_wildcards(obs_list, user_dict):
    """
    Loops through the user input and identifies wildcards in observation
    names and expands them.
    """
    new_dict = deepcopy(user_dict)
    for main_key, value in user_dict.items():
        new_entries = []
        non_matching_wildcards = []
        if main_key in ("UPDATE_KEYS", "CALCULATE_KEYS"):
            for val in value["keys"]:
                if "*" in val["key"]:
                    expanded = _expand_wildcard(obs_list, val, val["key"])
                    if len(expanded) == 0:
                        non_matching_wildcards.append(val["key"])
                    new_entries.extend(expanded)
                else:
                    new_entries.append(val)
            new_dict[main_key]["keys"] = new_entries

            if len(non_matching_wildcards) > 0:
                raise ValidationError(
                    f"Invalid {main_key}",
                    [f"{wc} had no match" for wc in non_matching_wildcards],
                )
    return new_dict


def create_active_lists(enkf_observations, events):
    """
    Will add observation vectors to observation data. Returns
    a list of tuples mirroring the user config but also containing
    the active list where the scaling factor will be applied.
    """
    new_events = []
    for event in events:
        obs_index = _data_index_to_obs_index(enkf_observations, event.key, event.index)
        new_active_list = _get_active_list(obs_index)

        new_events.append(_make_tuple(event.key, event.index, new_active_list))

    return new_events


def _get_active_list(index_list):
    """
    If the user doesn't supply an index list, we assume ALL-ACTIVE on the
    observations, otherwise an active list is
    created from the index list.
    """
    if index_list is None:
        index_list = []
    return _active_list_from_index_list(index_list)


def _make_tuple(
    key,
    index,
    active_list,
    new_event=namedtuple("named_dict", ["key", "index", "active_list"]),
):
    return new_event(key, index, active_list)


def _active_list_from_index_list(index_list):
    """
    Creates an ActiveList from a list of indexes
    :param index_list: list of index
    :type index_list:  list
    :return: Active list, a c-object with mode (ALL-ACTIVE, PARTIALLY-ACTIVE)
        and list of indices
    :rtype: active_list
    """
    active_list = ActiveList()
    for index in index_list:
        active_list.addActiveIndex(index)
    return active_list


def keys_with_data(observations, keys, ensemble_size, storage):
    """
    Checks that all keys have data and returns a list of error messages
    """
    active_realizations = storage.realizationList(RealizationStateEnum.STATE_HAS_DATA)

    if len(active_realizations) == 0:
        return []

    active_mask = BoolVector.createFromList(ensemble_size, active_realizations)
    return [key for key in keys if observations[key].hasData(active_mask, storage)]


def _data_index_to_obs_index(obs, obs_key, data_index_list):
    if obs[obs_key].getImplementationType().name != "GEN_OBS":
        return data_index_list
    if data_index_list is None:
        return data_index_list

    for timestep in obs[obs_key].getStepList():
        node = obs[obs_key].getNode(timestep)
        index_map = {node.getIndex(nr): nr for nr in range(len(node))}
    return [index_map[index] for index in data_index_list]
