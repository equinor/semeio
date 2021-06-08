import yaml
import pathlib
from copy import copy

# from functools import partial

# import cwrap
# from res.enkf import ErtScript
from res.enkf.enums.ert_impl_type_enum import ErtImplType
from res.enkf.enums.enkf_var_type_enum import EnkfVarType

# from res.enkf.enums.active_mode_enum import ActiveMode

# from res.enkf import EnkfObs
# from ecl.grid import EclRegion, EclGrid
# from ecl.eclfile import EclKW, Ecl3DKW
# from ecl.ecl_type import EclDataType
# from ecl.util.util import IntVector
# from ert_shared.plugins.plugin_manager import hook_implementation


# Global variables
debug_level = 1


def debug_print(text, level=debug_level):
    if level > 0:
        print(text)


def get_observations_from_ert(ert):
    # Get list of observation nodes from ERT config to be used in consistency check
    ert_obs = ert.getObservations()
    obs_data_from_ert_config = ert_obs.getAllActiveLocalObsdata()
    ert_obs_node_list = []
    for obs_node in obs_data_from_ert_config:
        node_name = obs_node.key()
        ert_obs_node_list.append(node_name)
    return ert_obs_node_list


def get_param_from_ert(ert, impl_type=ErtImplType.GEN_KW):
    ens_config = ert.ensembleConfig()
    keylist = ens_config.alloc_keylist()
    parameters_for_node = {}
    implementation_type_not_scalar = [
        ErtImplType.GEN_DATA,
        ErtImplType.FIELD,
        ErtImplType.SURFACE,
    ]

    for key in keylist:
        node = ens_config.getNode(key)
        impl_type = node.getImplementationType()
        var_type = node.getVariableType()
        if var_type == EnkfVarType.PARAMETER:
            if impl_type == ErtImplType.GEN_KW:
                # Node contains scalar parameters defined by GEN_KW
                kw_config_model = node.getKeywordModelConfig()
                string_list = kw_config_model.getKeyWords()
                param_list = []
                for name in string_list:
                    param_list.append(name)
                parameters_for_node[key] = param_list

            elif impl_type in implementation_type_not_scalar:
                # Node contains parameter from GEN_PARAM
                # which is of type PARAMETER
                # and implementation type GEN_DATA
                parameters_for_node[key] = None
    return parameters_for_node


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
            "does not match any observation or observation group specified."
        )
    return total_obs


def check_and_expand_param_groups(
    user_param_group, all_param_group_names, all_param_groups
):
    #  Check if user_param_group can be expanded to one or more param group names.
    # If so, get model parameters for these groups and return this as a list
    list_from_user_param_groups = [
        name
        for name in all_param_group_names
        if pathlib.Path(name).match(user_param_group)
    ]
    print(f"list_from_usr_param_groups: {list_from_user_param_groups}")
    total_param = []
    if len(list_from_user_param_groups) > 0:
        # Get all parameter names in all parameter groups found
        for param_group_name in list_from_user_param_groups:
            param_list = all_param_groups[param_group_name]
            total_param.extend(param_list)
    else:
        raise ValueError(
            f" The user defined parameter group {user_param_group} "
            "does not match any parameter, parameter node or parameter group specified."
        )
    return total_param


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


def get_full_parameter_name_list(user_node_name, all_param_dict):
    param_list = all_param_dict[user_node_name]
    full_param_name_list = []
    if param_list is not None:
        # A list of parameters for the specified node name exist
        for param_name in param_list:
            full_param_name = user_node_name.strip() + ":" + param_name.strip()
            full_param_name_list.append(full_param_name)
    else:
        # This parameter node does not have any named parameters since it can be
        # either a GEN_PARAM parameter node, a SURFACE parameter node
        # or a FIELD parameter node. Use the parameter node name instead.
        full_param_name_list.append(user_node_name)
    return full_param_name_list


def check_and_append_parameter_list(
    param,
    total_param,
    all_param_groups,
    all_param_dict,
    list_of_expanded_user_param_names,
    list_of_expanded_user_param_node_names,
):
    if all_param_groups is None:
        # No specification of user defined model parameter groups
        if len(list_of_expanded_user_param_names) > 0:
            # param is found in defined parameter names in ert instance
            total_param.extend(list_of_expanded_user_param_names)
        elif len(list_of_expanded_user_param_node_names) > 0:
            # param is not found in defined parameter names in ert instance
            # but it matches parameter node names found in ert instance
            for user_node_name in list_of_expanded_user_param_node_names:
                full_param_name_list = get_full_parameter_name_list(
                    user_node_name, all_param_dict
                )
                total_param.extend(full_param_name_list)
        else:
            # The expanded name param is not found in defined parameter names
            # or parameter node names.
            raise ValueError(
                f" The user defined parameter {param} "
                "does not match any parameter name or node in ert instance.\n"
                f"ERT instance nodes with parameters are: {all_param_dict}"
            )
    else:
        # In addition to check against defined parameter names and
        # defined parameter node names, also check against
        # defined parameter group names
        if len(list_of_expanded_user_param_names) > 0:
            total_param.extend(list_of_expanded_user_param_names)
        elif len(list_of_expanded_user_param_node_names) > 0:
            for user_node_name in list_of_expanded_user_param_node_names:
                full_param_name_list = get_full_parameter_name_list(
                    user_node_name, all_param_dict
                )
                total_param.extend(full_param_name_list)
        else:
            # Does not match defined parameters,
            # check with parameter groups instead
            all_param_group_names = []
            print(f"All param groups: {all_param_groups}")
            if isinstance(all_param_groups, dict):
                all_param_group_names = all_param_groups.keys()
            param_list_from_groups = check_and_expand_param_groups(
                param, all_param_group_names, all_param_groups
            )
            total_param.extend(param_list_from_groups)

    return total_param


def expand_wildcards_for_param(user_param, all_param_dict, all_param_groups=None):
    if len(all_param_dict) == 0:
        raise ValueError(" all_param_dict has 0 parameters")
    # All parameter names defined in ert instance
    all_param_list = []
    all_param_node_list = []
    for node_name in all_param_dict.keys():
        full_param_name_list = get_full_parameter_name_list(node_name, all_param_dict)
        all_param_list.extend(full_param_name_list)
        all_param_node_list.append(node_name)
    total_param = []
    tmp_input = copy(user_param)
    if not isinstance(user_param, list):
        user_param = [tmp_input]

    for param in user_param:
        if param.upper().strip() == "ALL":
            total_param.extend(all_param_list)
        else:
            # Find all defined parameters matching param
            list_of_expanded_user_param_names = [
                name for name in all_param_list if pathlib.Path(name).match(param)
            ]
            list_of_expanded_user_param_node_names = [
                name for name in all_param_node_list if pathlib.Path(name).match(param)
            ]
            total_param = check_and_append_parameter_list(
                param,
                total_param,
                all_param_groups,
                all_param_dict,
                list_of_expanded_user_param_names,
                list_of_expanded_user_param_node_names,
            )

    param_list = list(set(total_param))
    param_list.sort()
    return param_list


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

    if debug_level > 0:
        print(" -- Node names for observations found in ERT configuration:")
        for node_name in ert_obs_node_list:
            print(f"      {node_name}")
            print("\n")
    ert_obs_node_list.sort()
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
                raise KeyError(
                    f"Unknown keyword {keyword_read} found in {main_keyword}."
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
    obs_group_keyword = "obs_group"
    obs_keyword_items = correlation_spec_item[obs_group_keyword]
    valid_keywords = ["add", "remove"]
    mandatory_keywords = ["add"]

    # Check that mandatory keywords exists
    keywords = obs_keyword_items.keys()
    for mkey in mandatory_keywords:
        if mkey not in keywords:
            raise KeyError(
                f"Can not find mandatory keywords {mandatory_keywords} "
                f"in {obs_group_keyword} in {main_keyword} "
            )
        for keyword_read in keywords:
            if keyword_read not in valid_keywords:
                raise KeyError(
                    f"Unknown keyword {keyword_read} found in {obs_group_keyword} "
                    f"in {main_keyword}. "
                )
        add_list = obs_keyword_items["add"]
        remove_list = None
        if "remove" in keywords:
            remove_list = obs_keyword_items["remove"]
            if not isinstance(remove_list, list):
                remove_list = [remove_list]

        # Define the list of all observations to add including the obs groups used
        total_obs_names_added = expand_wildcards_for_obs(
            add_list, ert_obs_list, obs_group_dict
        )
        obs_node_names_added = list(set(total_obs_names_added))
        obs_node_names_added.sort()

        # Define the list of all observations to remove including the obs groups used
        if isinstance(remove_list, list):
            total_obs_names_removed = expand_wildcards_for_obs(
                remove_list, ert_obs_list, obs_group_dict
            )
            obs_node_names_removed = list(set(total_obs_names_removed))
            obs_node_names_removed.sort()

            # For each entry in add_list expand it to get observation names
            # For each entry in remove_list expand it to get observation names
            # Remove entries from add_list found in remove_list
            for name in obs_node_names_removed:
                if name in obs_node_names_added:
                    obs_node_names_added.remove(name)

    return obs_node_names_added


def read_param_groups(ert_param_dict, all_kw):
    main_keyword = "model_param_groups"
    param_group_item_list = all_kw[main_keyword]
    param_groups = {}
    for param_group_item in param_group_item_list:
        valid_keywords = ["add", "remove", "name"]
        mandatory_keywords = ["add", "name"]
        # Check that mandatory keywords exists
        keywords = param_group_item.keys()
        for mkey in mandatory_keywords:
            if mkey not in keywords:
                raise KeyError(
                    f"Can not find mandatory keywords {mandatory_keywords} "
                    f"in {main_keyword} "
                )
        for keyword_read in keywords:
            if keyword_read not in valid_keywords:
                raise KeyError(
                    f"Unknown keyword {keyword_read} found in {main_keyword}."
                )
        param_group_name = param_group_item["name"]
        param_groups[param_group_name] = []
        add_list = param_group_item["add"]
        remove_list = []
        if "remove" in keywords:
            remove_list = param_group_item["remove"]

        # For each entry in add_list expand it to get parameter names.
        # For each entry in remove_list expand it to get parameter names.
        # Remove entries from add_list found in remove_list.
        print(f"add_list before expand: {add_list}")
        print(f"remove_list before expand: {remove_list}")
        param_names_added = expand_wildcards_for_param(add_list, ert_param_dict)
        param_names_removed = expand_wildcards_for_param(remove_list, ert_param_dict)
        for name in param_names_removed:
            if name in param_names_added:
                param_names_added.remove(name)
        param_groups[param_group_name] = param_names_added
    return param_groups


def read_param_groups_for_correlations(
    param_group_dict, correlation_spec_item, main_keyword, ert_param_dict
):
    param_group_keyword = "model_group"
    param_keyword_items = correlation_spec_item[param_group_keyword]
    valid_keywords = ["add", "remove"]
    mandatory_keywords = ["add"]

    # Check that mandatory keywords exists
    keywords = param_keyword_items.keys()
    for mkey in mandatory_keywords:
        if mkey not in keywords:
            raise KeyError(
                f"Can not find mandatory keywords {mandatory_keywords} "
                f"in {param_group_keyword} in {main_keyword} "
            )
        for keyword_read in keywords:
            if keyword_read not in valid_keywords:
                raise KeyError(
                    f"Unknown keyword {keyword_read} "
                    f"found in {param_group_keyword} in {main_keyword}. "
                )
        add_list = param_keyword_items["add"]
        remove_list = None
        if "remove" in keywords:
            remove_list = param_keyword_items["remove"]
            if not isinstance(remove_list, list):
                remove_list = [remove_list]

        # Define the list of all observations to add including the obs groups used
        total_param_names_added = expand_wildcards_for_param(
            add_list, ert_param_dict, param_group_dict
        )
        param_names_added = list(set(total_param_names_added))
        param_names_added.sort()

        # Define the list of all parameters to remove including
        # the parameter groups used
        if isinstance(remove_list, list):
            total_param_names_removed = expand_wildcards_for_param(
                remove_list, ert_param_dict, param_group_dict
            )
            param_names_removed = list(set(total_param_names_removed))
            param_names_removed.sort()

            # For each entry in add_list expand it to get param names
            # For each entry in remove_list expand it to get param names
            # Remove entries from add_list found in remove_list
            for name in param_names_removed:
                if name in param_names_added:
                    param_names_added.remove(name)

    return param_names_added


def read_correlation_specification(
    all_kw, ert_obs_list, ert_param_dict, obs_groups=None, param_groups=None
):
    main_keyword = "correlations"
    correlation_spec_list = all_kw[main_keyword]
    correlation_dict = {}
    for corr_item in correlation_spec_list:
        corr_dict = {}
        name = corr_item["name"]
        obs_list = read_obs_groups_for_correlations(
            obs_groups, corr_item, main_keyword, ert_obs_list
        )
        corr_dict["obs_list"] = obs_list

        param_list = read_param_groups_for_correlations(
            param_groups, corr_item, main_keyword, ert_param_dict
        )
        corr_dict["param_list"] = param_list
        correlation_dict[name] = corr_dict
    return correlation_dict


def print_correlation_specification(correlation_specification, debug_level=1):
    if debug_level > 0:
        for name, item in correlation_specification.items():
            obs_list = item["obs_list"]
            param_list = item["param_list"]
            print(f" -- {name}")
            count = 0
            outstring = " "
            print(" -- Obs:")
            for obs_name in obs_list:
                outstring = outstring + "  " + obs_name
                count = count + 1
                if count > 9:
                    print(f"      {outstring}")
                    outstring = " "
                    count = 0
            if count > 0:
                print(f"      {outstring}")

            count = 0
            outstring = " "
            print(" -- Parameter:")
            for param_name in param_list:
                outstring = outstring + "  " + param_name
                count = count + 1
                if count > 4:
                    print(f"      {outstring}")
                    outstring = " "
                    count = 0
            if count > 0:
                print(f"      {outstring}")
            print("\n")
        print("\n")


def active_index_for_parameter(full_param_name, ert_param_dict):
    # For parameters defined as scalar parameters (coming from GEN_KW)
    # which is of the form  node_name:parameter_name,
    # split the node_name from the parameter_name to identify which one
    # of the parameter_names are used (and should be active).
    # For parameters that are not coming from GEN_KW,
    # the active index is not used here.
    contains_both_node_and_param_name = False
    for c in full_param_name:
        if c == ":":
            contains_both_node_and_param_name = True
            break
    if contains_both_node_and_param_name:
        [node_name, param_name] = full_param_name.split(":")

        param_list = ert_param_dict[node_name]
        index = -1
        for count, name in enumerate(param_list):
            if name == param_name:
                index = count
                break
        assert index > -1
    else:
        node_name = full_param_name
        param_name = full_param_name
        index = None
    return node_name, param_name, index


def check_for_duplicated_correlation_specifications(correlation_dict):
    # All observations and model parameters used in correlations
    tmp_obs_list = []
    tmp_param_list = []
    for name, item in correlation_dict.items():
        obs_list = item["obs_list"]
        param_list = item["param_list"]
        tmp_obs_list.extend(obs_list)
        tmp_param_list.extend(param_list)
    complete_obs_list = list(set(tmp_obs_list))
    complete_obs_list.sort()
    complete_param_list = list(set(tmp_param_list))
    complete_param_list.sort()

    # Initialize the table
    correlation_table = {}
    for obs_name in complete_obs_list:
        for param_name in complete_param_list:
            key = (obs_name, param_name)
            correlation_table[key] = False

    # Check each correlation set (corresponding to a ministep)
    number_of_duplicates = 0
    for name, item in correlation_dict.items():
        obs_list = item["obs_list"]
        param_list = item["param_list"]
        for obs_name in obs_list:
            for param_name in param_list:
                key = (obs_name, param_name)
                if correlation_table[key]:
                    debug_print(
                        f"-- When reading correlation: {name} there are "
                        f"double specified correlations for {key}",
                        1,
                    )
                    number_of_duplicates = number_of_duplicates + 1
                else:
                    correlation_table[key] = True
    return number_of_duplicates


def add_ministeps(correlation_specification, ert_param_dict, ert):
    local_config = ert.getLocalConfig()
    updatestep = local_config.getUpdatestep()
    count = 1
    for ministep_name, item in correlation_specification.items():
        ministep = local_config.createMinistep(ministep_name)

        model_group_name = ministep_name + "_param_group_" + str(count)
        model_param_group = local_config.createDataset(model_group_name)

        obs_group_name = ministep_name + "_obs_group_" + str(count)
        obs_group = local_config.createObsdata(obs_group_name)

        obs_list = item["obs_list"]
        param_list = item["param_list"]
        count = count + 1

        # Setup model parameter group
        node_names_used = []
        for full_param_name in param_list:
            node_name, param_name, index = active_index_for_parameter(
                full_param_name, ert_param_dict
            )
            if node_name not in node_names_used:
                model_param_group.addNode(node_name)
                node_names_used.append(node_name)

            if index is not None:
                #    This is a model parameter node from GEN_KW
                active_param_list = model_param_group.getActiveList(node_name)
                active_param_list.addActiveIndex(index)

        # Setup observation group
        for obs_name in obs_list:
            obs_group.addNode(obs_name)

        # Setup ministep
        debug_print(f" -- Attach {model_group_name} to ministep {ministep_name}")
        ministep.attachDataset(model_param_group)

        debug_print(f" -- Attach {obs_group_name} to ministep {ministep_name}")
        ministep.attachObsset(obs_group)

        debug_print(f" -- Add {ministep_name} to update step\n")
        updatestep.attachMinistep(ministep)


def clear_correlations(ert):
    local_config = ert.getLocalConfig()
    local_config.clear()
