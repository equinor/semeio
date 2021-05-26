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


def check_and_expand_obs_groups(user_obs_group, all_obs_group_names, all_obs_groups):
    #  Check of user_obs_group can be expanded to one or more obs group names.
    # If so, get all observations for these groups and return this as a list
    list_from_user_obs_groups = [
        name for name in all_obs_group_names if pathlib.Path(name).match(user_obs_group)
    ]
    total_obs = []
    if len(list_from_user_obs_groups) > 0:
        # Get all obs in all obs groups found
        for obs_group_name in list_from_user_obs_groups:
            obs_list = all_obs_groups[obs_group_name]
            total_obs.extend(obs_list)
    else:
        raise ValueError(
            f" The user defined observation group {user_obs_group} "
            "does not match any observation group specified."
        )
    return total_obs


def expand_wildcards_for_obs(user_obs, all_observations, all_obs_groups=None):
    if len(all_observations) == 0:
        raise ValueError(" all_observations has 0 observations")
    total_obs = []
    tmp_input = copy(user_obs)
    if not isinstance(user_obs, list):
        user_obs = [tmp_input]

    all_obs_group_names = []
    if isinstance(all_obs_groups, dict):
        all_obs_group_names = all_obs_groups.keys()

    for obs in user_obs:
        if obs.upper().strip() == "ALL":
            total_obs.extend(all_observations)
            debug_print(f" -- Expand ALL: {total_obs}")
        else:
            # Find all defined observations matching obs
            list_from_user_obs = [
                name for name in all_observations if pathlib.Path(name).match(obs)
            ]
            if all_obs_groups is None:
                # Require that obs match defined observations
                if len(list_from_user_obs) > 0:
                    total_obs.extend(list_from_user_obs)
                else:
                    raise ValueError(
                        f" The user defined observations {obs} "
                        "does not match any observations in ert instance."
                    )
            else:
                # Require that obs match defined observations or defined obs groups
                if len(list_from_user_obs) > 0:
                    total_obs.extend(list_from_user_obs)
                else:
                    # Does not match defined observations, check with obs groups instead
                    obs_list_from_groups = check_and_expand_obs_groups(
                        obs, all_obs_group_names, all_obs_groups
                    )
                    total_obs.extend(obs_list_from_groups)

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


def read_obs_groups(ert_obs_list, all_kw):
    #   ert_obs = ert.getObservations()
    #   obs_data_from_ert_config = ert_obs.getAllActiveLocalObsdata()
    #   ert_obs_list = get_obs_from_ert_config(obs_data_from_ert_config)

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

        # For each entry in add_list expand it to get observation names.
        # For each entry in remove_list expand it to get observation names.
        # If obs group names are found instead of obs names,
        # get the obs names for the group.
        # Remove entries from add_list found in remove_list.
        obs_node_names_added = expand_wildcards_for_obs(add_list, ert_obs_list)
        obs_node_names_removed = expand_wildcards_for_obs(remove_list, ert_obs_list)
        for name in obs_node_names_removed:
            if name in obs_node_names_added:
                obs_node_names_added.remove(name)
        obs_groups[obs_group_name] = obs_node_names_added
    return obs_groups


def read_obs_groups_for_correlations(
    obs_group_dict, correlation_spec_item, main_keyword, ert_obs_list
):
    all_defined_obs_groups_list = obs_group_dict.keys()
    debug_print(
        f"-- All defined obs groups from keyword 'obs_groups' \n"
        f"     {all_defined_obs_groups_list}"
    )

    obs_group_keyword = "obs_group"
    obs_keyword_items = correlation_spec_item[obs_group_keyword]
    valid_keywords = ["add", "remove"]
    mandatory_keywords = ["add"]

    # Check that mandatory keywords exists
    keywords = obs_keyword_items.keys()
    debug_print(f" -- keywords: {keywords}")
    for mkey in mandatory_keywords:
        if mkey not in keywords:
            raise KeyError(
                f"Can not find mandatory keywords {mandatory_keywords} "
                f"in {obs_group_keyword} in {main_keyword} "
            )
        for keyword_read in keywords:
            if keyword_read not in valid_keywords:
                print(
                    f" -- Warning: Unknown keyword {keyword_read} "
                    f"found in {obs_group_keyword} in {main_keyword}. "
                    "Ignored"
                )
        add_list = obs_keyword_items["add"]
        remove_list = None
        if "remove" in keywords:
            remove_list = obs_keyword_items["remove"]
            if not isinstance(remove_list, list):
                remove_list = [remove_list]
        debug_print(f"add_list: {add_list}")
        debug_print(f"remove_list: {remove_list}")

        # Define the list of all observations to add including the obs groups used
        total_obs_names_added = expand_wildcards_for_obs(
            add_list, ert_obs_list, obs_group_dict
        )
        obs_node_names_added = list(set(total_obs_names_added))
        obs_node_names_added.sort()
        debug_print(f" -- obs node names added: {obs_node_names_added}")

        # Define the list of all observations to remove including the obs groups used
        if isinstance(remove_list, list):
            total_obs_names_removed = expand_wildcards_for_obs(
                remove_list, ert_obs_list, obs_group_dict
            )
            obs_node_names_removed = list(set(total_obs_names_removed))
            obs_node_names_removed.sort()
            debug_print(f" -- obs node names removed: {obs_node_names_removed}")

            # For each entry in add_list expand it to get observation names
            # For each entry in remove_list expand it to get observation names
            # Remove entries from add_list found in remove_list
            for name in obs_node_names_removed:
                if name in obs_node_names_added:
                    obs_node_names_added.remove(name)

    return obs_node_names_added
