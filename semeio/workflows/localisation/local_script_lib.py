import yaml
import pathlib
import math
import copy


from res.enkf.enums.ert_impl_type_enum import ErtImplType
from res.enkf.enums.enkf_var_type_enum import EnkfVarType

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


def get_param_from_ert(ert):
    ens_config = ert.ensembleConfig()
    keylist = ens_config.alloc_keylist()
    parameters_for_node = {}
    grid_config = None
    implementation_type_not_scalar = [
        ErtImplType.GEN_DATA,
        ErtImplType.FIELD,
        ErtImplType.SURFACE,
    ]
    dimensions = None
    grid_origo = None
    rotation_angle = None
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
                # Node contains parameter from FIELD
                # No named parameters for FIELD, but number of indexed parameters
                # are equal to nx*ny*nz for ERTBOX grids.
                parameters_for_node[key] = []
                if impl_type == ErtImplType.FIELD and dimensions is None:
                    grid = ert.eclConfig().getGrid()
                    dimensions = grid.get_dims()
                    bounding_box = grid.get_bounding_box_2d()
                    grid_origo, rotation_angle, area_size = grid_info(bounding_box)
                    grid_config = {}
                    grid_config["dimensions"] = dimensions
                    grid_config["grid_origo"] = grid_origo
                    grid_config["rotation"] = rotation_angle
                    grid_config["size"] = area_size
                elif impl_type == ErtImplType.GEN_DATA:
                    gen_data_config = node.getModelConfig()
                    data_size = gen_data_config.getDataSize()
                    parameters_for_node[key] = data_size

    return parameters_for_node, grid_config


def grid_info(bounding_box):
    # Corner points bounding_box = (p1, p2, p3, p4)
    p1 = bounding_box[0]
    #    p2 = bounding_box[1]
    p3 = bounding_box[2]
    p4 = bounding_box[3]

    grid_origo = p4
    # axis for I direction

    V1x = p3[0] - p4[0]
    V1y = p3[1] - p4[1]
    V1_norm = math.sqrt(V1x * V1x + V1y * V1y)
    #    print(f"V1: ({V1x},{V1y})")
    V2x = p4[0] - p1[0]
    V2y = p4[1] - p1[1]
    V2_norm = math.sqrt(V2x * V2x + V2y * V2y)
    #    print(f"V2: ({V2x},{V2y})")

    sintheta = V1y / V1_norm
    theta = math.asin(sintheta)

    if V1y >= 0:
        if V1x >= 0:
            #   0<= theta <= pi/2
            pass
        else:
            #  pi/2 <= theta <= pi
            theta = math.pi + theta
    else:
        if V1x >= 0:
            # 3pi/2 <= theta <= 2pi
            theta = 2 * math.pi + theta
        else:
            #    pi <= theta <= 3pi/2
            theta = theta + math.pi

    rotation_angle = theta * 180 / math.pi
    grid_size = (V1_norm, V2_norm)
    return grid_origo, rotation_angle, grid_size


def expand_wildcards(patterns, list_of_words):
    all_matches = []
    errors = []
    for pattern in patterns:
        matches = [
            words for words in list_of_words if pathlib.Path(words).match(pattern)
        ]
        if len(matches) > 0:
            all_matches.extend(matches)
        else:
            # No match
            errors.append(pattern)
    all_matches = list(set(all_matches))
    all_matches.sort()
    if len(errors) > 0:
        raise ValueError(
            " These observation specifications does not match "
            "any observations defined in ERT model\n"
            f"     {errors}"
        )

    return all_matches


def get_full_parameter_name_list(user_node_name, all_param_dict):
    param_list = all_param_dict[user_node_name]
    full_param_name_list = []
    if len(param_list) > 0:
        for param_name in param_list:
            full_param_name = user_node_name.strip() + ":" + param_name.strip()
            full_param_name_list.append(full_param_name)
    else:
        # This node does not have any specified list of parameters.
        # There are two possibilities:
        #  - The node may have parameters with names (coming from GEN_KW keyword in ERT)
        #  - The node may not have any parameters with names
        #     (coming from GEN_PARAM, SURFACE, FIELD) keywords.
        full_param_name_list.append(user_node_name)
    return full_param_name_list


def full_param_names_for_ert(ert_param_dict):
    all_param_node_list = ert_param_dict.keys()
    valid_full_param_names = []
    #    print(f"all_param_node_list: {all_param_node_list}")
    for node_name in all_param_node_list:
        # Define full parameter name of form nodename:parametername to simplify matching
        full_name_list_for_node = get_full_parameter_name_list(
            node_name, ert_param_dict
        )
        valid_full_param_names.extend(full_name_list_for_node)
    return valid_full_param_names


def set_full_param_name(node_name, param_name):
    full_name = node_name + ":" + param_name
    return full_name


def append_pattern_list(pattern_of_dict, pattern_list):
    for node_name_pattern, param_name_pattern_list in pattern_of_dict.items():
        if isinstance(param_name_pattern_list, list):
            for param_name_pattern in param_name_pattern_list:
                full_name_pattern = set_full_param_name(
                    node_name_pattern, param_name_pattern
                )
                pattern_list.append(full_name_pattern)

                if param_name_pattern == "*":
                    # A node with * as an entry in the parameter list may be
                    # a node without named parameters like nodes created by
                    # the FIELD, SURFACE, GEN_PARAM ert keywords.
                    # In this case the full name should also include only the
                    # node name to get a match with valid full parameter names
                    full_name_pattern = node_name_pattern.strip()
                    pattern_list.append(full_name_pattern)
        elif isinstance(param_name_pattern_list, str):
            # Only a single string as value for the node in the dict
            full_name_pattern = set_full_param_name(
                node_name_pattern.strip(), param_name_pattern_list.strip()
            )
            pattern_list.append(full_name_pattern)
    return pattern_list


def pattern_list_from_dict(list_of_pattern_of_dicts):
    pattern_list = []
    for pattern_of_dict in list_of_pattern_of_dicts:
        if isinstance(pattern_of_dict, dict):
            pattern_list = append_pattern_list(pattern_of_dict, pattern_list)
        if isinstance(pattern_of_dict, str):
            pattern_list.append(pattern_of_dict)
        else:
            raise TypeError(f"Expecting a string , get a {pattern_of_dict}")
    return pattern_list


def split_full_param_names_and_define_dict(full_param_name_list):
    # Split the name up into node name and param name and define a dict
    param_dict = {}
    param_list = None
    for fullname in full_param_name_list:
        #        print(f"fullname: {fullname}")
        words = []
        words = fullname.split(":")
        #        print(f"words: {words}")
        if len(words) == 1:
            node_name = words[0]
            param_name = None
        else:
            node_name = words[0]
            param_name = words[1]
        #        print(f" node_name: {node_name}  param_name: {param_name}")
        if node_name not in param_dict.keys():
            param_dict[node_name] = []
        param_list = param_dict[node_name]
        if param_name is not None:
            param_list.append(param_name)
    return param_dict


def expand_wildcards_with_param_dicts(list_of_pattern_of_dicts, ert_param_dict):
    # Define a list with full parameter names of the form nodename:paramname
    valid_full_param_names = full_param_names_for_ert(ert_param_dict)
    #    print(f"valid full param names: {valid_full_param_names}")

    # Define a list with specfied patterns of parameter names of the
    # form nodename:paramname where both nodename and paramname
    # can be specified with wildcard notation
    pattern_list = pattern_list_from_dict(list_of_pattern_of_dicts)
    #    print(f"pattern_list: {pattern_list}")

    # Expand patterns of full parameter names
    list_of_all_expanded_full_names = []
    for pattern in pattern_list:
        #        print(f"pattern: {pattern}")
        expanded_full_names = [
            name for name in valid_full_param_names if pathlib.Path(name).match(pattern)
        ]
        #        print(f"expanded: {expanded_full_names}")
        list_of_all_expanded_full_names.extend(expanded_full_names)

    # Eliminate duplicates
    full_param_name_list = list(set(list_of_all_expanded_full_names))
    full_param_name_list.sort()
    #    print(f"full param name list: {full_param_name_list}")

    # Make a dict with specified node names as key and a list of
    # parameter names as value
    param_dict = split_full_param_names_and_define_dict(full_param_name_list)
    print(f"param_dict: {param_dict}")
    return param_dict


def expand_wildcards_with_param(pattern_list, ert_param_dict):
    # Define a list with full parameter names of the form nodename:paramname
    valid_full_param_names = full_param_names_for_ert(ert_param_dict)
    #    print(f"valid full param names: {valid_full_param_names}")

    # Expand patterns of full parameter names
    list_of_all_expanded_full_names = []
    error_list = []
    for pattern in pattern_list:
        #        print(f"pattern: {pattern}")
        expanded_full_names = [
            name for name in valid_full_param_names if pathlib.Path(name).match(pattern)
        ]
        if len(expanded_full_names) > 0:
            #            print(f"expanded: {expanded_full_names}")
            list_of_all_expanded_full_names.extend(expanded_full_names)
        else:
            error_list.append(pattern)

    if len(error_list) > 0:
        raise ValueError(
            "List of specified model parameters "
            "that does not match any ERT parameter:\n"
            f"     {error_list}"
        )

    # Eliminate duplicates
    full_param_name_list = list(set(list_of_all_expanded_full_names))
    full_param_name_list.sort()
    #    print(f"full param name list: {full_param_name_list}")

    # Make a dict with specified node names as key and a list of
    # parameter names as value
    param_dict = split_full_param_names_and_define_dict(full_param_name_list)
    #  print(f"param_dict: {param_dict}")

    return param_dict


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


def active_index_for_parameter(node_name, param_name, ert_param_dict):
    # For parameters defined as scalar parameters (coming from GEN_KW)
    # the parameters for a node have a name. Getthe index from the orderrrrr
    # of these parameters.
    # For parameters that are not coming from GEN_KW,
    # the active index is not used here.

    ert_param_list = ert_param_dict[node_name]
    if len(ert_param_list) > 0:
        index = -1
        for count, name in enumerate(ert_param_list):
            if name == param_name:
                index = count
                break
        assert index > -1
    else:
        index = None
    return index


def check_for_duplicated_correlation_specifications(correlations):
    # All observations and model parameters used in correlations
    #    print(f"correlations: {correlations}")
    tmp_obs_list = []
    tmp_param_list = []
    for corr in correlations:
        obs_list = corr.obs_group.add
        tmp_obs_list.extend(copy.copy(obs_list))

        param_dict = corr.param_group.add
        for node_name in param_dict.keys():
            full_param_name_list = get_full_parameter_name_list(node_name, param_dict)
            tmp_param_list.extend(copy.copy(full_param_name_list))
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
    for corr in correlations:
        name = corr.name
        obs_list = corr.obs_group.add
        param_dict = corr.param_group.add

        for obs_name in obs_list:
            for node_name in param_dict.keys():
                full_param_name_list = get_full_parameter_name_list(
                    node_name, param_dict
                )
                for param_name in full_param_name_list:
                    key = (obs_name, param_name)
                    if key not in correlation_table.keys():
                        raise KeyError(
                            " Correlation_table does not have the key:"
                            f"({obs_name},{param_name})"
                        )
                    if correlation_table[key]:
                        debug_print(
                            f"-- When reading correlation: {name} there are "
                            f"double specified correlations for {key}",
                            0,
                        )
                        number_of_duplicates = number_of_duplicates + 1
                    else:
                        correlation_table[key] = True
    return number_of_duplicates


def add_ministeps(user_config, ert_param_dict, ert):
    local_config = ert.getLocalConfig()
    updatestep = local_config.getUpdatestep()
    ens_config = ert.ensembleConfig()
    grid_for_field = None
    for count, corr_spec in enumerate(user_config.correlations):
        ministep_name = corr_spec.name
        ministep = local_config.createMinistep(ministep_name)

        param_group_name = ministep_name + "_param_group"
        model_param_group = local_config.createDataset(param_group_name)

        obs_group_name = ministep_name + "_obs_group"
        obs_group = local_config.createObsdata(obs_group_name)

        obs_list = corr_spec.obs_group.add
        param_dict = corr_spec.param_group.add

        # Setup model parameter group
        for node_name, param_list in param_dict.items():
            node = ens_config.getNode(node_name)
            impl_type = node.getImplementationType()

            debug_print(f"  -- Add node: {node_name} of type: {impl_type}")
            model_param_group.addNode(node_name)
            if impl_type == ErtImplType.GEN_KW:
                active_param_list = model_param_group.getActiveList(node_name)
                for param_name in param_list:
                    index = active_index_for_parameter(
                        node_name, param_name, ert_param_dict
                    )
                    if index is not None:
                        debug_print(f" -- Active parameter index: {index}")
                        active_param_list.addActiveIndex(index)
            elif impl_type == ErtImplType.FIELD:
                if corr_spec.field_scale is not None:
                    # A scaling of correlation is defined
                    debug_print(
                        " -- Apply the correlation scaling method: "
                        f"{corr_spec.field_scale.method}"
                    )
                    if grid_for_field is None:
                        grid_for_field = ert.eclConfig().getGrid()
                        debug_print(f" -- Get grid: {grid_for_field.get_name()}")

                    row_scaling = model_param_group.row_scaling(node_name)
                    field_config = node.getFieldModelConfig()
                    data_size = field_config.get_data_size()
                    if corr_spec.field_scale.method == "gaussian_decay":
                        ref_pos = corr_spec.obs_group.ref_point
                        main_range = corr_spec.field_scale.method_param[0]
                        perp_range = corr_spec.field_scale.method_param[1]
                        azimuth = corr_spec.field_scale.method_param[2]
                        row_scaling.assign(
                            data_size,
                            GaussianDecay(
                                ref_pos, main_range, perp_range, azimuth, grid_for_field
                            ),
                        )
                    elif corr_spec.field_scale.method == "exponential_decay":
                        ref_pos = corr_spec.obs_group.ref_point
                        main_range = corr_spec.field_scale.method_param[0]
                        perp_range = corr_spec.field_scale.method_param[1]
                        azimuth = corr_spec.field_scale.method_param[2]
                        row_scaling.assign(
                            data_size,
                            ExponentialDecay(
                                ref_pos, main_range, perp_range, azimuth, grid_for_field
                            ),
                        )
                    else:
                        print(
                            f" --  Scaling method: {corr_spec.field_scale.method} "
                            "is not implemented yet"
                        )
        # Setup observation group
        for obs_name in obs_list:
            debug_print(f"Add obs node: {obs_name}")
            obs_group.addNode(obs_name)

        # Setup ministep
        debug_print(f" -- Attach {param_group_name} to ministep {ministep_name}")
        ministep.attachDataset(model_param_group)

        debug_print(f" -- Attach {obs_group_name} to ministep {ministep_name}")
        ministep.attachObsset(obs_group)

        debug_print(f" -- Add {ministep_name} to update step\n")
        updatestep.attachMinistep(ministep)


def clear_correlations(ert):
    local_config = ert.getLocalConfig()
    local_config.clear()


# def get_config_path(ert)
#    pass
#    return None


def check_ref_point_with_grid_data(ref_point, origo, rotation, size):
    #    print(f"size: {size}, origo: {origo}, ref_point: {ref_point}, "
    #    f"rotation: {rotation}")
    point = transform_from_utm_to_ertbox_coordinates(ref_point, origo, rotation)
    if not (0 <= point[0] <= size[0] and 0 <= point[1] <= size[1]):
        raise ValueError(
            f"Specified reference point ({ref_point[0]}, {ref_point[1]}) "
            "for observations is outside ERTBOX area.\n"
            "The reference point transformed to ERTBOX "
            "coordinates fails the check:\n"
            f"1.coordinate: 0 <= {point[0]} <= {size[0]}      "
            f"2.coordinate:  0<= {point[1]} <= {size[1]}"
        )


def transform_from_utm_to_ertbox_coordinates(point, origo, rotation):
    theta = rotation * math.pi / 180.0
    sintheta = math.sin(theta)
    costheta = math.cos(theta)
    x0 = point[0] - origo[0]
    y0 = point[1] - origo[1]
    x = costheta * x0 + sintheta * y0
    y = -sintheta * x0 + costheta * y0
    transf_point = [x, y]
    #    print(f"Point: ({point[0]},{point[1]})  Transf point: ({x},{y})")
    return transf_point


def transform_from_ertbox_to_utm_coordinates(point, origo, rotation):
    theta = rotation * math.pi / 180.0
    sintheta = math.sin(theta)
    costheta = math.cos(theta)
    x0 = point[0]
    y0 = point[1]
    x = costheta * x0 - sintheta * y0
    y = sintheta * x0 + costheta * y0
    x = x + origo[0]
    y = y + origo[1]

    transf_point = [x, y]
    print(f"Point: ({point[0]},{point[1]})  Transf point: ({x},{y})")
    return transf_point


# This is an example callable which decays as a gaussian away from a position
# obs_pos; this is (probaly) a simplified version of tapering function which
# will be interesting to use.
class GaussianDecay(object):
    def __init__(self, ref_point, main_range, perp_range, azimuth, grid):
        self._obs_pos = ref_point
        self._main_range = main_range
        self._perp_range = perp_range
        self._grid = grid

        angle = (90.0 - azimuth) * math.pi / 180.0
        self._cosangle = math.cos(angle)
        self._sinangle = math.sin(angle)

    def __call__(self, data_index):
        x, y, _ = self._grid.get_xyz(active_index=data_index)
        x_unrotated = x - self._obs_pos[0]
        y_unrotated = y - self._obs_pos[1]

        dx = (
            x_unrotated * self._cosangle + y_unrotated * self._sinangle
        ) / self._main_range
        dy = (
            -x_unrotated * self._sinangle + y_unrotated * self._cosangle
        ) / self._perp_range
        exp_arg = -0.5 * (dx * dx + dy * dy)
        return math.exp(exp_arg)


class ExponentialDecay(object):
    def __init__(self, ref_point, main_range, perp_range, azimuth, grid):
        self._obs_pos = ref_point
        self._main_range = main_range
        self._perp_range = perp_range
        self._grid = grid

        angle = (90.0 - azimuth) * math.pi / 180.0
        self._cosangle = math.cos(angle)
        self._sinangle = math.sin(angle)

    def __call__(self, data_index):
        x, y, _ = self._grid.get_xyz(active_index=data_index)
        x_unrotated = x - self._obs_pos[0]
        y_unrotated = y - self._obs_pos[1]

        dx = (
            x_unrotated * self._cosangle + y_unrotated * self._sinangle
        ) / self._main_range
        dy = (
            -x_unrotated * self._sinangle + y_unrotated * self._cosangle
        ) / self._perp_range
        exp_arg = -0.5 * math.sqrt(dx * dx + dy * dy)
        return math.exp(exp_arg)
