# Localisation configuration script
# Written by: O.Lia
# Date: April 2021
# Current version:
#   Localisation configuration file in YAML format is read for scalar
#   parameters and this script setup the localisation in ERT.


import yaml

# from functools import partial

# import cwrap
from res.enkf import ErtScript
from res.enkf.enums.ert_impl_type_enum import ErtImplType

# from res.enkf import EnkfObs
# from ecl.grid import EclRegion, EclGrid
# from ecl.eclfile import EclKW, Ecl3DKW
# from ecl.ecl_type import EclDataType
# from ecl.util.util import IntVector
from ert_shared.plugins.plugin_manager import hook_implementation


# Global variables
debug_level = 1


def debug_print(text):
    if debug_level > 0:
        print(text)


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


def get_param_nodes_from_gen_kw(ert):
    ens_config = ert.ensembleConfig()
    keylist = ens_config.getKeylistFromImplType(ErtImplType.GEN_KW)
    parameters_for_node = {}
    for key in keylist:
        node = ens_config.getNode(key)
        impl_type = node.getImplementationType()
        if impl_type == ErtImplType.GEN_KW:
            kw_config_model = node.getKeywordModelConfig()
            parameters_for_node[key] = kw_config_model.getKeyWords()
    return parameters_for_node


def get_list_item_name(name, item_list):
    item = None
    key = "name"
    for item in item_list:
        if key in item.keys():
            if name == item[key]:
                return item
    return item


def set_active_parameters_for_node(
    model_group,
    node_name,
    specified_active_parameter_dict,
    defined_scalar_parameter_nodes,
    invert_active=False,
):
    """
    Go through the list of specified parameter names for this node,
    check that it is defined and set the parameters active or inactive.
    There are two modes, normal where invert_active = False
    and inverse mode where invert_active = True. Normal node is
    used when adding active correlations and inverse mode
    when removing active correlations.
    """
    active_param_list = model_group.getActiveList(node_name)
    defined_parameter_names = defined_scalar_parameter_nodes[node_name]
    for pname, isActive in specified_active_parameter_dict.items():
        index = -1
        for count, param_defined in enumerate(defined_parameter_names):
            if pname == param_defined:
                index = count
                break
        if index == -1:
            raise ValueError(
                f"Parameter with name: {pname} is not defined for node: "
                f"{node_name} in keyword 'defined_scalar_parameters' \n"
                f"Check specification of model parameter group "
                f"{model_group.name()} or node definition in keyword  "
                f" 'defined_scalar_parameters' "
            )
        if (invert_active and (not isActive)) or ((not invert_active) and isActive):
            debug_print(
                f" ---- Set parameter: {pname} in node {node_name} in "
                f"model group {model_group.name()} active\n"
                f" ---- Add active index:  {index}"
            )
            active_param_list.addActiveIndex(index)


def initialize_model_group_scalar(
    local_config,
    model_group_name,
    mini_step_name,
    node_names_for_group_list,
    defined_scalar_parameter_nodes,
    remove_nodes=False,
):
    """
    For each node in model parameter group, check that the node is defined in the
    ERT configuration.  For each model parameter in the node,
    check that the model parameter is defined in the ERT configuration.
    Set active/inactive the model parameters for this node
    according to localisation configuration.
    Only initialize the model parameter group if it is not already initialized.
    """
    if remove_nodes:
        # Create a group containing all model parameters
        all_reduced_name = "ALL_EXCEPT_" + model_group_name + "_" + mini_step_name
        debug_print(
            f" -- Create model group {all_reduced_name} containing "
            "ALL model parameter"
        )
        all_reduced = local_config.copyDataset("ALL_DATA", all_reduced_name)

        for item in node_names_for_group_list:
            # Check that it is defined in ERT before initializing it
            node_name = item["name"]
            if node_name not in defined_scalar_parameter_nodes.keys():
                raise ValueError(
                    f"The node with name: {node_name}  found under model param group "
                    f"{model_group_name}\n"
                    f"is  not defined in keyword: 'defined_scalar_parameters' "
                )
            # Remove the model parameter nodes for the current model parameter group
            debug_print(f" -- Remove node {node_name} from {all_reduced_name}")
            del all_reduced[node_name]
            all_active = item["all_active"]
            if all_active:
                # This means that all model parameters for this node should be removed
                # from correlation. Nothing more to do here
                pass
            else:
                # Only a sub set of all model parameters for this node should be
                # removed from correlation
                debug_print(
                    f" -- Add node {node_name} with selected active "
                    f"parameters to {all_reduced_name}"
                )
                all_reduced.addNode(node_name)
                specified_active_param_dict = item["active"]
                set_active_parameters_for_node(
                    all_reduced,
                    node_name,
                    specified_active_param_dict,
                    defined_scalar_parameter_nodes,
                    invert_active=True,
                )

        model_param_group = all_reduced
        new_model_group_name = all_reduced_name
    else:
        # Create new group
        debug_print(f" -- Create model parameter group: {model_group_name}")
        model_param_group = local_config.createDataset(model_group_name)
        new_model_group_name = model_group_name
        for item in node_names_for_group_list:
            node_name = item["name"]
            # Check that it is defined in ERT before initializing it
            if node_name not in defined_scalar_parameter_nodes.keys():
                raise ValueError(
                    f"The node with name: {node_name}  found under model param group "
                    f"{model_group_name}\n"
                    f"is  not defined in keyword: 'defined_scalar_parameters' "
                )
            all_active = item["all_active"]
            debug_print(f" -- Add node: {node_name} in group: {model_group_name}")
            model_param_group.addNode(node_name)

            if all_active:
                debug_print(f" -- All parameters for node {node_name} are active")
            else:
                debug_print(
                    f" -- Some or no parameters for node {node_name} are active"
                )
            if not all_active:
                specified_active_param_dict = item["active"]
                set_active_parameters_for_node(
                    model_param_group,
                    node_name,
                    specified_active_param_dict,
                    defined_scalar_parameter_nodes,
                    invert_active=False,
                )

    return model_param_group, new_model_group_name


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


def define_obs_group(
    local_config,
    obs_group_name,
    obs_groups_list,
    ert_obs_node_list,
):
    # Create new obs group
    obs_group = local_config.createObsdata(obs_group_name)
    obs_group_item = get_list_item_name(obs_group_name, obs_groups_list)
    if obs_group_item is None:
        raise ValueError(
            f"Observation group {obs_group_name} " "is not found in 'obs_groups'"
        )
    obs_nodes_list = obs_group_item["nodes"]
    for obs_node_name in obs_nodes_list:
        debug_print(f" -- Add node: {obs_node_name} in  group: {obs_group_name}")

        # Check that obs node is defined in ERT config and add node
        if obs_node_name in ert_obs_node_list:
            obs_group.addNode(obs_node_name)
        else:
            raise ValueError(
                f"Obs node name {obs_node_name} is not defined in ERT config file\n"
            )
    return obs_group


def remove_obs_groups(all_obs_groups, mini_step_list, obs_groups_list):
    """
    Remove all obs_groups containing nodes that have some or all model parameters
    not to be correlated to any model parameter group
    """
    removed_node_name_list = []
    for ministep_item in mini_step_list:
        obs_group_name = ministep_item["obs_group"]
        obs_group_item = get_list_item_name(obs_group_name, obs_groups_list)
        if obs_group_item is None:
            raise ValueError(
                f"Observation group {obs_group_name} is not defined in 'obs_groups'"
            )

        obs_nodes_list = obs_group_item["nodes"]
        for obs_node_name in obs_nodes_list:
            if obs_node_name not in removed_node_name_list:
                debug_print(
                    f" -- Remove obs node {obs_node_name} from "
                    f"{all_obs_groups.name()}"
                )
                removed_node_name_list.append(obs_node_name)
                del all_obs_groups[obs_node_name]


def create_ministep(local_config, mini_step_name, model_group, obs_group, updatestep):
    ministep = local_config.createMinistep(mini_step_name)
    debug_print(f" -- Attach {model_group.name()} to ministep {mini_step_name}")
    ministep.attachDataset(model_group)

    debug_print(f" -- Attach {obs_group.name()} to ministep {mini_step_name}")
    ministep.attachObsset(obs_group)

    debug_print(f" -- Add {mini_step_name} to update step\n")
    updatestep.attachMinistep(ministep)


def localisation_setup_add_correlations(
    ert, all_kw, defined_scalar_parameter_nodes, defined_ert_obs_list
):

    local_config = ert.getLocalConfig()

    # Clear all correlations
    local_config.clear()
    updatestep = local_config.getUpdatestep()

    # Loop over all mini step specifications and setup the ministeps
    mini_step_list = all_kw["add_correlations"]
    model_param_group_list = all_kw["model_param_groups"]
    obs_groups_list = all_kw["obs_groups"]
    model_param_group_used = {}
    obs_group_used = {}
    for ministep_item in mini_step_list:
        mini_step_name = ministep_item["name"]
        model_group_name = ministep_item["model_group"]
        obs_group_name = ministep_item["obs_group"]
        debug_print(
            f" -- Get specification of ministep: {mini_step_name}\n"
            f" -- Use model group:  {model_group_name}  and obs group: "
            f"{obs_group_name}"
        )

        # Define model_group
        model_param_group = None
        if model_group_name not in model_param_group_used.keys():
            model_param_group_item = get_list_item_name(
                model_group_name, model_param_group_list
            )
            if model_param_group_item is None:
                raise ValueError(
                    f"Model parameter group {model_group_name} "
                    "is not defined in 'model_param_groups'"
                )
            group_type = model_param_group_item["type"]
            if group_type == "Scalar":
                node_names_for_group_list = model_param_group_item["nodes"]
                # model_param_group is assigned the specified nodes and
                # parameters for each node is set active or inactive
                model_param_group, new_model_group_name = initialize_model_group_scalar(
                    local_config,
                    model_group_name,
                    mini_step_name,
                    node_names_for_group_list,
                    defined_scalar_parameter_nodes,
                    remove_nodes=False,
                )

            else:
                raise ValueError("Nodes that are not scalars are not yet implemented")

            model_param_group_used[model_group_name] = model_param_group
        else:
            model_param_group = model_param_group_used[model_group_name]

        # Define obs group
        if obs_group_name not in obs_group_used.keys():
            obs_group = define_obs_group(
                local_config, obs_group_name, obs_groups_list, defined_ert_obs_list
            )
            obs_group_used[obs_group_name] = obs_group
        else:
            obs_group = obs_group_used[obs_group_name]

        # Define mini step object
        create_ministep(
            local_config, mini_step_name, model_param_group, obs_group, updatestep
        )


def localisation_setup_remove_correlations_old_and_wrong_version(
    ert, all_kw, defined_scalar_parameter_nodes, defined_ert_obs_list
):
    """
    In this case the specification in keyword 'remove_correlations' is interpreted
    as correlations to remove.
    Algorithm:
    1. Identify all observation nodes involved in correlations to be removed
    2. For each observation:
    3.   Create one obs group per observation which should not be correlated
          with ALL model parameters, but possibly with some model parameters.
          Hence, one group for each observation that is a part of the correlations
          to be removed. Create one obs group containing all the other observations.
          These are observations that should be correlated with ALL model parameters
    4. Define a ministep for all observations that is correlated to ALL
        model parameters.
    5. For each observation group except the one for observations used in
        step 4 above:
           Identify all model parameters the observations should not correlate to.
           Make a model parameter group that is a copy of the set of all model
           parameters. From this group, remove the model parameters the
           observation group should not be correlated to.
           Specify a ministep for the correlation between the model
           parameters remaining and the observation group.
           Repeat step 5 for all observations that should not be correlated to ALL
           model parameters.

    """
    local_config = ert.getLocalConfig()

    # Clear all correlations
    local_config.clear()
    updatestep = local_config.getUpdatestep()

    # Loop over all correlations to be removed and create a list of the
    # observation nodes
    mini_step_list = all_kw["remove_correlations"]
    model_param_group_list = all_kw["model_param_groups"]
    obs_groups_list = all_kw["obs_groups"]
    model_param_group_used = {}
    inverted_model_group_name = {}
    obs_group_used = {}
    for ministep_item in mini_step_list:
        mini_step_name = ministep_item["name"]
        model_group_name = ministep_item["model_group"]
        obs_group_name = ministep_item["obs_group"]
        debug_print(
            f"\n -- Get specification of correlation to remove: {mini_step_name}\n"
            f" -- Remove correlation between:  {model_group_name}  and "
            f"{obs_group_name}"
        )

        # Define model_group to correlate with obs group
        model_param_group = None
        if model_group_name not in model_param_group_used.keys():
            model_param_group_item = get_list_item_name(
                model_group_name, model_param_group_list
            )
            if model_param_group_item is None:
                raise ValueError(
                    f"Model parameter group {model_group_name} "
                    "is not defined in 'model_param_groups'"
                )
            group_type = model_param_group_item["type"]
            if group_type == "Scalar":
                node_names_for_group_list = model_param_group_item["nodes"]
                # model_param_group is assigned the specified nodes and
                # parameters for each node is set active or inactive
                model_param_group, new_model_group_name = initialize_model_group_scalar(
                    local_config,
                    model_group_name,
                    mini_step_name,
                    node_names_for_group_list,
                    defined_scalar_parameter_nodes,
                    remove_nodes=True,
                )
                inverted_model_group_name[model_group_name] = new_model_group_name
            else:
                raise ValueError("Nodes that are not scalars are not yet implemented")

            model_param_group_used[model_group_name] = model_param_group
        else:
            new_model_group_name = inverted_model_group_name[model_group_name]
            debug_print(
                f" -- Re-use the already defined model group " f"{new_model_group_name}"
            )
            model_param_group = model_param_group_used[model_group_name]

        # Define obs group
        if obs_group_name not in obs_group_used.keys():
            obs_group = define_obs_group(
                local_config,
                obs_group_name,
                obs_groups_list,
                defined_ert_obs_list,
            )
            obs_group_used[obs_group_name] = obs_group
        else:
            obs_group = obs_group_used[obs_group_name]

        # Define mini step object
        create_ministep(
            local_config, mini_step_name, model_param_group, obs_group, updatestep
        )

    all_obs_groups_name = "ALL_OBS_GROUPS"
    debug_print(f" -- Create obs group {all_obs_groups_name} with all obs nodes")
    all_obs_groups = local_config.copyObsdata("ALL_OBS", all_obs_groups_name)

    all_model_groups_name = "ALL_MODEL_GROUPS"
    debug_print(
        f" -- Create model group  {all_model_groups_name} "
        f"with all model parameter nodes"
    )
    all_model_groups = local_config.copyDataset("ALL_DATA", all_model_groups_name)

    # Remove all obs_groups containing nodes that have some or all model parameters
    # not to be correlated to any model parameter group
    remove_obs_groups(all_obs_groups, mini_step_list, obs_groups_list)

    # Define mini step object to correlate all model parameters with all obs that is not
    # specified to have no correlations with model parameters
    if len(all_obs_groups) > 0:
        mini_step_name = "ALL_CORRELATION_REDUCED"
        create_ministep(
            local_config, mini_step_name, all_model_groups, all_obs_groups, updatestep
        )
    else:
        debug_print(" -- No observations are correlated to all model parameters")


def get_obs_groups_with_reduced_correlations(remove_correlation_spec):
    """
    Loop over all correlations specified to be removed.
    Find which observation groups don't have correlations to all
    model parameter groups, but only to some or no model parameter groups.
    Return a dictionary with observation group name as key and a list of
    model parameter groups the observation group is not or only parly correlated to.
    """
    remove_correlation_dict = {}
    for item in remove_correlation_spec:
        correlation_name = item["name"]
        model_group_name = item["model_group"]
        obs_group_name = item["obs_group"]
        debug_print(
            f"\n -- Get specification of correlation to remove: {correlation_name}\n"
            f" -- Use model group:  {model_group_name}  and obs group: "
            f"{obs_group_name}"
        )
        list_of_model_groups_per_obs_group = []
        if obs_group_name in remove_correlation_dict.keys():
            list_of_model_groups_per_obs_group = remove_correlation_dict[obs_group_name]
            list_of_model_groups_per_obs_group.append(model_group_name)
        else:
            list_of_model_groups_per_obs_group.append(model_group_name)
            remove_correlation_dict[obs_group_name] = list_of_model_groups_per_obs_group
    return remove_correlation_dict


def create_ministep_for_obs_and_model_groups_with_full_correlation(
    remove_correlation_spec, obs_groups_list, local_config, updatestep
):
    """
    All observation groups that have correlations to all model parameters are
    defined as one observation group. This observation group is attached to a ministep
    together with a model parameter group containing all model parameters.
    These observations are all the observations not specified in
    keyword 'remove_correlations' but specified in 'obs_groups'.
    """
    all_obs_groups_name = "ALL_OBS_GROUPS"
    debug_print(f" -- Create obs group {all_obs_groups_name} with all obs nodes")
    all_obs_groups = local_config.copyObsdata("ALL_OBS", all_obs_groups_name)

    all_model_groups_name = "ALL_MODEL_GROUPS"
    debug_print(
        f" -- Create model group  {all_model_groups_name} "
        f"with all model parameter nodes"
    )
    all_model_groups = local_config.copyDataset("ALL_DATA", all_model_groups_name)

    # Remove all obs_groups containing nodes that have some or all model parameters
    # not to be correlated to any model parameter group
    remove_obs_groups(all_obs_groups, remove_correlation_spec, obs_groups_list)

    # Define mini step object to correlate all model parameters with all obs that is not
    # specified to have no correlations with model parameters
    if len(all_obs_groups) > 0:
        mini_step_name = "ALL_CORRELATION_REDUCED"
        create_ministep(
            local_config, mini_step_name, all_model_groups, all_obs_groups, updatestep
        )
    else:
        debug_print(" -- No observations are correlated to all model parameters")


def localisation_setup_remove_correlations(
    ert, all_kw, defined_scalar_parameter_nodes, defined_ert_obs_list
):
    """
     In this case the specification in keyword 'remove_correlations' is interpreted
     as correlations to remove.
     Algorithm:
     1. Find all observation groups that are specified not to be correlated with some
         or all model parameter groups.
     2. For each such observation group, find all model parameter groups that are
         specified not to be correlated with the observation group and make a
         dictionary with observation group as key and a list of model parameter group
         names as the value.
     3. For each observation group look up and get the list of model parameter groups
     4.    Create a model parameter group ALL_MODEL_GROUPS as copy
            of ALL_DATA containing all model parameter nodes.
     5.    For each model parameter group in the list,
            get the list of model parameter nodes
     6.       For each model parameter node in the model parameter group
     7.            Delete the node from the ALL_MODEL_GROUPS
     8.            Add the node again if not all model parameters in the node
                    should be disabled from having correlation to the observation group.
     9.            Set active the model parameters in the node that should be active.
    10.    Attach the reduced model parameter group ALL_MODEL_GROUPS
             where the unwanted parameter nodes/variables are disabled or removed
             to a new ministep.
    11.    Attach the observation group to the same ministep.
    12.    Attach the ministep to the updatestep.
    13. Repeat from step 3 untill a ministep is defined for all observations
          having correlations with a reduced set of model parameter groups is defined.
    14. Define an observation group containing all the observations having
          correlations with all model parameter groups.
    15.  Attach this observation group to a new ministep.
    16.  Define a model parameter group containing ALL model parameter groups.
    17.  Attach this model parametr group to the same ministep.
    18.  Attach thus new ministep to the update step.

    """
    local_config = ert.getLocalConfig()

    # Clear all correlations
    local_config.clear()
    updatestep = local_config.getUpdatestep()

    remove_correlation_spec = all_kw["remove_correlations"]
    model_param_group_list = all_kw["model_param_groups"]
    obs_groups_list = all_kw["obs_groups"]

    # Loop over all correlations to be removed and find the observation groups
    # that are specified. Create a list of model parameter groups for
    # each observation group.
    remove_correlation_dict = get_obs_groups_with_reduced_correlations(
        remove_correlation_spec
    )

    # Remove the model parameter nodes from the object containing all nodes
    ministep_number = 1
    for obs_group_name, model_group_names in remove_correlation_dict.items():
        if debug_level > 0:
            print(
                f" -- List of model group nodes not to correlate to {obs_group_name}:"
            )
            for name in model_group_names:
                print(f"          {name}")

        # Create a group containing all model parameters
        all_reduced_name = "ALL_MODEL_GROUPS_NOT_CORRELATED_WITH_" + obs_group_name
        debug_print(
            f" -- Create model group {all_reduced_name} containing "
            "ALL model parameter"
        )
        all_reduced = local_config.copyDataset("ALL_DATA", all_reduced_name)
        for model_group_name in model_group_names:
            model_param_group_item = get_list_item_name(
                model_group_name, model_param_group_list
            )
            if model_param_group_item is None:
                raise ValueError(
                    f"Model parameter group {model_group_name} "
                    "is not defined in 'model_param_groups'"
                )
            # Get the list of model param nodes for this group
            group_type = model_param_group_item["type"]
            node_names_for_group_list = None
            if group_type == "Scalar":
                node_names_for_group_list = model_param_group_item["nodes"]
            else:
                raise ValueError("Nodes that are not scalars are not yet implemented")

            # Remove the node from the model param group with all
            # model parameter nodes
            for item in node_names_for_group_list:
                # Check that it is defined in ERT
                node_name = item["name"]
                if node_name not in defined_scalar_parameter_nodes.keys():
                    raise ValueError(
                        f"The node with name: {node_name}  found under "
                        f"model param group {model_group_name}\n"
                        f"is  not defined in keyword: 'defined_scalar_parameters' "
                    )
                debug_print(f" -- Remove node {node_name} from {all_reduced_name}")
                del all_reduced[node_name]
                all_active = item["all_active"]
                if all_active:
                    # This means that all model parameters for this node should be
                    # removed from correlation. Nothing more to do here
                    pass
                else:
                    # Only a sub set of all model parameters for this node should be
                    # removed from correlation
                    debug_print(
                        f" -- Add node {node_name} with selected active "
                        f"parameters to {all_reduced_name}"
                    )
                    all_reduced.addNode(node_name)
                    specified_active_param_dict = item["active"]
                    set_active_parameters_for_node(
                        all_reduced,
                        node_name,
                        specified_active_param_dict,
                        defined_scalar_parameter_nodes,
                        invert_active=True,
                    )

        model_param_group = all_reduced

        # Define obs group
        obs_group = define_obs_group(
            local_config,
            obs_group_name,
            obs_groups_list,
            defined_ert_obs_list,
        )

        # Define mini step object
        mini_step_name = "CORRELATION_" + str(ministep_number)
        create_ministep(
            local_config, mini_step_name, model_param_group, obs_group, updatestep
        )
        ministep_number = ministep_number + 1

    # All observation groups that have correlations to all model parameters are
    # defined as one observation group. This observation group is attached to a ministep
    # together with a model parameter group containing all model parameters.
    # These observations are all the observations not specified in
    # keyword 'remove_correlations' but specified in 'obs_groups'.
    create_ministep_for_obs_and_model_groups_with_full_correlation(
        remove_correlation_spec, obs_groups_list, local_config, updatestep
    )


class LocalisationConfigJob(ErtScript):
    def run(self, *args):
        ert = self.ert()

        # Read yml file with specifications
        all_kw = read_localisation_config(args)

        # Path to configuration setup
        #        config_path = all_kw["config_path"]

        # Get ministep specifications

        key1 = "add_correlations"
        key2 = "remove_correlations"
        if key1 in all_kw.keys() and key2 not in all_kw.keys():
            localisation_mode = "ADD"
            debug_print(" -- Correlation mode: Add correlations")
        elif key1 not in all_kw.keys() and key2 in all_kw.keys():
            localisation_mode = "REMOVE"
            debug_print(" -- Correlation mode: Remove correlations")
        else:
            raise KeyError(
                "Missing specification of either keyword 'add_correlation' or "
                "'remove_correlation' or both are specified. "
                "Only one of them should be used."
            )

        # Get a dictionary of node names with associated scalar parameter names from
        # ERT configuration if it is specified. This is specified by keyword
        # defined_scalar_parameters and the parameter  names are read from
        # the GEN_KW distribution file for the parameters.
        defined_scalar_parameter_nodes = get_param_nodes_from_gen_kw(ert)
        debug_print(" -- Parameter nodes from GEN_KW:")
        for node_name, parameter_list in defined_scalar_parameter_nodes.items():
            if debug_level > 0:
                print(f" -- Parameter node: {node_name}")
                print(" --   Parameter names:")
                for name in parameter_list:
                    print(f"           {name}")

        # Get list of observation nodes from ERT config to be used in consistency check
        ert_obs = ert.getObservations()
        obs_data_from_ert_config = ert_obs.getAllActiveLocalObsdata()
        defined_ert_obs_list = get_obs_from_ert_config(obs_data_from_ert_config)

        if localisation_mode == "ADD":
            localisation_setup_add_correlations(
                ert, all_kw, defined_scalar_parameter_nodes, defined_ert_obs_list
            )
        elif localisation_mode == "REMOVE":
            localisation_setup_remove_correlations(
                ert, all_kw, defined_scalar_parameter_nodes, defined_ert_obs_list
            )


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(LocalisationConfigJob, "LOCALISATION_JOB")
    workflow.description = "test"
    workflow.category = "observations.correlation"
