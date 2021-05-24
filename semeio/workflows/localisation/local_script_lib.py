import yaml
import pathlib
from copy import copy

# from functools import partial

# import cwrap
# from res.enkf import ErtScript
# from res.enkf.enums.ert_impl_type_enum import ErtImplType
# from res.enkf.enums.active_mode_enum import ActiveMode

# from res.enkf import EnkfObs
# from ecl.grid import EclRegion, EclGrid
# from ecl.eclfile import EclKW, Ecl3DKW
# from ecl.ecl_type import EclDataType
# from ecl.util.util import IntVector
# from ert_shared.plugins.plugin_manager import hook_implementation


# Global variables
debug_level = 1

# Dictionary with key =(obs_name, model_param_name) containing value
# True or False depending on whether the correlation is active or not.
# Use this to check that no double specification of a correlation appears.
correlation_table = {}


def debug_print(text):
    if debug_level > 0:
        print(text)


def expand_wildcards(user_obs, all_observations):
    total_obs = []
    tmp_input = copy(user_obs)
    if not isinstance(user_obs, list):
        user_obs = [tmp_input]
    debug_print(f" -- Input to expand_wildcard: {user_obs}")
    for obs in user_obs:
        print(f" obs: {obs}  {obs.upper().strip()}")
        if obs.upper().strip() == "ALL":
            total_obs.extend(all_observations)
            debug_print(f" -- Expand ALL: {total_obs}")
        else:
            total_obs.extend(
                [name for name in all_observations if pathlib.Path(name).match(obs)]
            )
    obs_list = list(set(total_obs))
    obs_list.sort()
    return obs_list


def read_localisation_config(args):
    if len(args) == 1:
        specification_file_name = args[0]
    else:
        raise ValueError(f"Expecting a single argument. Got {args} arguments.")

    print(
        "\n" f"Define localisation setup using config file: {specification_file_name}"
    )
    with open(specification_file_name, "r") as yml_file:
        localisation_yml = yaml.safe_load(yml_file)
    all_keywords = localisation_yml["localisation"]
    if all_keywords is None:
        raise IOError(f"None is found when reading file: {specification_file_name}")
    return all_keywords


def get_obs_from_ert_config(obs_data_from_ert_config):
    ert_obs_node_list = []
    for obs_node in obs_data_from_ert_config:
        node_name = obs_node.key()
        ert_obs_node_list.append(node_name)

    debug_print(" -- Node names for observations found in ERT configuration:")
    for node_name in ert_obs_node_list:
        debug_print(f"      {node_name}")
    debug_print("\n")
    return ert_obs_node_list


def read_obs_groups(ert, all_kw):
    ert_obs = ert.getObservations()
    obs_data_from_ert_config = ert_obs.getAllActiveLocalObsdata()
    ert_obs_list = get_obs_from_ert_config(obs_data_from_ert_config)

    main_keyword = "obs_groups"
    obs_group_item_list = all_kw[main_keyword]
    obs_groups = {}
    for obs_group_item in obs_group_item_list:
        valid_keywords = ["add", "remove", "name"]
        mandatory_keywords = ["add", "name"]
        # Check that mandatory keywords exists
        keywords = obs_group_item.keys()
        for mkey in mandatory_keywords:
            if mkey not in keywords:
                raise KeyError(
                    f"Can not find mandatory keywords {mandatory_keywords} "
                    f"in {main_keyword} "
                )
        for keyword_read in keywords:
            if keyword_read not in valid_keywords:
                print(
                    f" -- Warning: Unknown keyword {keyword_read} "
                    f"found in {main_keyword}. Ignored"
                )
        obs_group_name = obs_group_item["name"]
        obs_groups[obs_group_name] = []
        add_list = obs_group_item["add"]
        remove_list = []
        if "remove" in keywords:
            remove_list = obs_group_item["remove"]
        #        debug_print(f"obs_group_item: {obs_group_item}")
        #        debug_print(f"add_list: {add_list}")
        #        debug_print(f"remove_list: {remove_list}")

        # For each entry in add_list expand it to get observation names
        # For each entry in remove_list expand it to get observation names
        # Remove entries from add_list found in remove_list
        obs_node_names_added = expand_wildcards(add_list, ert_obs_list)
        obs_node_names_removed = expand_wildcards(remove_list, ert_obs_list)
        for name in obs_node_names_removed:
            if name in obs_node_names_added:
                obs_node_names_added.remove(name)
        obs_groups[obs_group_name] = obs_node_names_added
    return obs_groups
