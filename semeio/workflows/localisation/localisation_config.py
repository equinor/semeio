# pylint: disable=E0213
import copy
import pathlib
from typing import List, Optional, Union, Dict
from typing_extensions import Literal

from pydantic import BaseModel, validator, confloat
from semeio.workflows.localisation.localisation_debug_settings import (
    LogLevel,
    debug_print,
)
import semeio.workflows.localisation.localisation_debug_settings as log_level_setting
from res.enkf.enums.ert_impl_type_enum import ErtImplType


def split_full_param_names_and_define_dict(full_param_name_list):
    # Split the name up into node name and param name and define a dict
    param_dict = {}
    param_list = None
    for fullname in full_param_name_list:
        words = []
        words = fullname.split(":")
        if len(words) == 1:
            node_name = words[0]
            param_name = None
        else:
            node_name = words[0]
            param_name = words[1]
        if node_name not in param_dict.keys():
            param_dict[node_name] = []
        param_list = param_dict[node_name]
        if param_name is not None:
            param_list.append(param_name)
    return param_dict


def get_full_parameter_name_list(user_node_name, all_param_dict, node_type_dict=None):
    param_list = all_param_dict[user_node_name]
    full_param_name_list = []
    if node_type_dict is None:
        if len(param_list) > 0:
            for param_name in param_list:
                full_param_name = user_node_name.strip() + ":" + param_name.strip()
                full_param_name_list.append(full_param_name)
        else:
            full_param_name_list.append(user_node_name)
    else:
        if node_type_dict[user_node_name] == ErtImplType.GEN_KW:
            if len(param_list) > 0:
                for param_name in param_list:
                    full_param_name = user_node_name.strip() + ":" + param_name.strip()
                    full_param_name_list.append(full_param_name)
            else:
                full_param_name_list.append(user_node_name)

        elif node_type_dict[user_node_name] == ErtImplType.GEN_DATA:
            data_size = int(param_list[0])
            for index in range(data_size):
                full_param_name = user_node_name.strip() + ":" + str(index)
                full_param_name_list.append(full_param_name)

        elif node_type_dict[user_node_name] == ErtImplType.FIELD:
            full_param_name_list.append(user_node_name)

        elif node_type_dict[user_node_name] == ErtImplType.SURFACE:
            full_param_name_list.append(user_node_name)

        # This node does not have any specified list of parameters.
        # There are two possibilities:
        #  - The node may have parameters with names
        #     (coming from GEN_KW keyword in ERT)
        #  - The node may not have any parameters with names
        #     (coming from GEN_PARAM, SURFACE, FIELD) keywords.

    return full_param_name_list


def set_full_param_name(node_name, param_name):
    full_name = node_name + ":" + param_name
    return full_name


def full_param_names_for_ert(ert_param_dict, ert_node_type):
    all_param_node_list = ert_param_dict.keys()
    valid_full_param_names = []
    for node_name in all_param_node_list:
        # Define full parameter name of form nodename:parametername to
        # simplify matching.
        full_name_list_for_node = get_full_parameter_name_list(
            node_name, ert_param_dict, ert_node_type
        )
        valid_full_param_names.extend(full_name_list_for_node)
    return valid_full_param_names


def check_ref_point_with_surface_data(surface_scale, ref_point):
    debug_print(
        f"No check is implemented to verify that ref point {ref_point}\n"
        f"      is within surface area defined by {surface_scale.filename}",
        LogLevel.LEVEL3,
    )


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


def expand_wildcards_with_param(pattern_list, ert_param_dict, ert_node_type):
    # Define a list with full parameter names of the form nodename:paramname
    valid_full_param_names = full_param_names_for_ert(ert_param_dict, ert_node_type)

    # Expand patterns of full parameter names
    list_of_all_expanded_full_names = []
    error_list = []
    for pattern in pattern_list:
        expanded_full_names = [
            name for name in valid_full_param_names if pathlib.Path(name).match(pattern)
        ]
        if len(expanded_full_names) > 0:
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

    # Make a dict with specified node names as key and a list of
    # parameter names as value
    param_dict = split_full_param_names_and_define_dict(full_param_name_list)

    return param_dict


def check_for_duplicated_correlation_specifications(correlations):
    # All observations and model parameters used in correlations
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
                            f"When reading correlation: {name} there are "
                            f"double specified correlations for {key}",
                            LogLevel.LEVEL3,
                        )
                        number_of_duplicates = number_of_duplicates + 1
                    else:
                        correlation_table[key] = True
    return number_of_duplicates


class ObsConfig(BaseModel):
    add: Union[str, List[str]]
    remove: Optional[Union[str, List[str]]]

    @validator("add")
    def validate_add(cls, add):
        if isinstance(add, str):
            add = [add]
        return add

    @validator("remove")
    def validate_remove(cls, remove):
        if isinstance(remove, str):
            remove = [remove]
        return remove


class ParamConfig(BaseModel):
    add: Union[str, List[str]]
    remove: Optional[Union[str, List[str]]]

    @validator("add")
    def validate_add(cls, add):
        if isinstance(add, str):
            add = [add]
        return add

    @validator("remove")
    def validate_remove(cls, remove):
        if isinstance(remove, str):
            remove = [remove]
        return remove


class GaussianConfig(BaseModel):
    method: Literal["gaussian_decay"]
    main_range: confloat(gt=0)
    perp_range: confloat(gt=0)
    angle: confloat(ge=0.0, le=360)
    filename: Optional[str]


class ExponentialConfig(BaseModel):
    method: Literal["exponential_decay"]
    main_range: confloat(gt=0)
    perp_range: confloat(gt=0)
    angle: confloat(ge=0, le=360)
    filename: Optional[str]


class CorrelationConfig(BaseModel):
    name: str
    obs_group: ObsConfig
    param_group: ParamConfig
    ref_point: Optional[List[float]]
    field_scale: Optional[Union[GaussianConfig, ExponentialConfig]]
    surface_scale: Optional[Union[GaussianConfig, ExponentialConfig]]

    @validator("ref_point", pre=True)
    def validate_ref_point(cls, value):
        ref_point = value
        if len(ref_point) == 2:
            ref_point[0] = float(ref_point[0])
            ref_point[1] = float(ref_point[1])
        else:
            raise ValueError(
                "Reference point for an observation group "
                "must be a list of two float or integer. "
                f"Found the reference point: {ref_point}"
            )
        return ref_point

    @validator("field_scale", pre=True)
    def validate_field_scale(cls, value):
        """
        To improve the user feedback we explicitly check
        which method is configured and bypass the Union
        """
        if isinstance(value, BaseModel):
            return value
        if not isinstance(value, dict):
            raise ValueError("value must be dict")
        method = value.get("method")
        _valid_methods = {
            "gaussian_decay": GaussianConfig,
            "exponential_decay": ExponentialConfig,
        }
        if method in _valid_methods.keys():
            return _valid_methods[method](**value)
        else:
            raise ValueError(
                f"Unknown method: {method}, valid methods are: {_valid_methods.keys()}"
            )

    @validator("surface_scale", pre=True)
    def validate_surface_scale(cls, value):
        """
        To improve the user feedback we explicitly check
        which method is configured and bypass the Union
        """
        if isinstance(value, BaseModel):
            return value
        if not isinstance(value, dict):
            raise ValueError("value must be dict")
        key = "filename"
        if key in value.keys():
            surface_file_name = value.get(key)
        else:
            raise KeyError(f"Missing keyword {key} in keyword 'surface_scale' ")
        if not isinstance(surface_file_name, str):
            raise ValueError(f"Invalid file name for surface: {surface_file_name}")
        method = value.get("method")
        _valid_methods = {
            "gaussian_decay": GaussianConfig,
            "exponential_decay": ExponentialConfig,
        }
        if method in _valid_methods.keys():
            return _valid_methods[method](**value)
        else:
            raise ValueError(
                f"Unknown method: {method}, valid methods are: {_valid_methods.keys()}"
            )


class LocalisationConfig(BaseModel):
    """
    observations:  A list of observations from ERT in format nodename
    parameters:    A dict of  parameters from ERT in format nodename:paramname.
                            Key is node name. Values are lists of parameter names
                            for the node.
    node_type:     A dict with node type from ERT.
                            Key is node name.
                            Values are implementation type for the node.
    correlations:   A list of CorrelationConfig objects keeping name of
                            one correlation set which defines the input to
                            create a ministep object.
    log_level:       Integer defining how much log output to write to screen
    """

    observations: List[str]
    parameters: Dict[str, Optional[Union[str, List[str]]]]
    node_type: Optional[Dict[str, int]]
    correlations: List[CorrelationConfig]
    log_level: Optional[Union[str, int]]

    @validator("log_level")
    def validate_log_level(cls, value):
        valid_log_levels = [
            LogLevel.OFF,
            LogLevel.LEVEL1,
            LogLevel.LEVEL2,
            LogLevel.LEVEL3,
            LogLevel.LEVEL4,
        ]
        level = int(value)
        if level not in valid_log_levels:
            print(
                f"Specified logging level {level} is unknown. \n"
                f"Must be integer with {LogLevel.OFF} <= value <= {LogLevel.LEVEL4}"
            )
            level = LogLevel.OFF

        # Change the log level from default to user defined
        log_level_setting.debug_level = level

    @validator("correlations")
    def validate_correlations(cls, correlations, values):

        for corr in correlations:
            if len(corr.obs_group.add) == 0:
                raise ValueError(
                    "Number of specified observations for correlation: "
                    f"{corr.name} is 0"
                )
            observations = check_observation_specification(
                corr.obs_group, values["observations"]
            )
            corr.obs_group.add = observations
            corr.obs_group.remove = []

            node_params_dict = values["parameters"]
            node_type_dict = None
            if "node_type" in values.keys():
                node_type_dict = values["node_type"]
            parameters = get_and_validate_parameters(
                corr.param_group, node_params_dict, node_type_dict
            )
            corr.param_group.add = parameters
            corr.param_group.remove = []

            # Check if any parameter is a field and check if ref_point is defined.
            # If the ref_point is defined it should be within the area defined
            # by the 3D grid defined for the field. This is however not checked here but
            # is postponed to the function defining the ministeps
            check_validation_of_ref_point_for_field(corr, node_type_dict, values)

            # Check if any parameter is a surface and check if ref_point is defined.
            # If the ref_point is defined, it should be within the area defined
            # by the 2D grid defined for the surface.
            check_validation_of_ref_point_for_surface(corr, node_type_dict, values)

        number_of_duplicates = check_for_duplicated_correlation_specifications(
            correlations
        )
        if number_of_duplicates > 0:
            raise ValueError(
                "Some correlations are specified multiple times. "
                f"Found {number_of_duplicates} duplicated correlations."
            )

        return correlations


def check_observation_specification(obs_group, ert_obs_list):
    obs_added = expand_wildcards(obs_group.add, ert_obs_list)
    if obs_group.remove is not None:
        obs_removed = expand_wildcards(obs_group.remove, ert_obs_list)
        observations = [obs for obs in obs_added if obs not in obs_removed]
    else:
        observations = obs_added
    return observations


def check_validation_of_ref_point_for_field(corr, node_type_dict, values):
    param_group_dict = corr.param_group.add
    field_scale = corr.field_scale
    ref_point = corr.ref_point

    for node_name, param_list in param_group_dict.items():
        if node_type_dict[node_name] == ErtImplType.FIELD:
            if field_scale is not None:
                # Require that ref_point must be defined
                if ref_point is None:
                    raise KeyError(
                        "When using FIELD, the reference point must be specified."
                    )


def check_validation_of_ref_point_for_surface(corr, node_type_dict, values):
    param_group_dict = corr.param_group.add
    surface_scale = corr.surface_scale
    ref_point = corr.ref_point
    for node_name, param_list in param_group_dict.items():
        if node_type_dict[node_name] == ErtImplType.SURFACE:
            if surface_scale is not None:
                # Require that ref_point must be defined
                if ref_point is None:
                    raise KeyError(
                        "When using SURFACE or FIELD, "
                        "the reference point must be specified."
                    )

                # Check that ref_point is within area defined by SURFACE
                check_ref_point_with_surface_data(surface_scale, ref_point)


def get_and_validate_parameters(param_group, values, node_type):
    params_added = expand_wildcards_with_param(param_group.add, values, node_type)
    if param_group.remove is not None:
        params_removed = expand_wildcards_with_param(
            param_group.remove, values, node_type
        )
        nodes_to_remove = []
        for node_name, param_names in params_added.items():
            if node_name in params_removed:
                param_names_to_remove = params_removed[node_name]
                param_names_in_node_after_remove = [
                    param for param in param_names if param not in param_names_to_remove
                ]
                params_added[node_name] = param_names_in_node_after_remove
                if len(param_names_in_node_after_remove) == 0:
                    nodes_to_remove.append(node_name)
        for node_name in nodes_to_remove:
            del params_added[node_name]
        parameters = params_added
    else:
        parameters = params_added

    return parameters
