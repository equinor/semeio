from typing import List, Optional, Union, Dict

from pydantic import BaseModel, validator
from semeio.workflows.localisation.local_script_lib import (
    expand_wildcards,
    expand_wildcards_with_param,
    check_for_duplicated_correlation_specifications,
    check_ref_point_with_grid_data,
)


class ObsConfig(BaseModel):
    add: Union[str, List[str]]
    remove: Optional[Union[str, List[str]]]
    ref_point: Optional[List[Union[float, int]]]

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

    @validator("ref_point")
    def validate_ref_point(cls, ref_point):
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


class FieldConfig(BaseModel):
    method: str
    method_param: List[Union[float, int]]


class CorrelationConfig(BaseModel):
    name: str
    obs_group: ObsConfig
    param_group: ParamConfig
    field_scale: Optional[FieldConfig]


class GridInfoConfig(BaseModel):
    grid_dim: List[Union[float, int]]
    origo: List[Union[float, int]]
    grid_rotation: Union[float, int]
    size: List[Union[float, int]]


class LocalisationConfig(BaseModel):
    """
    observations:  A list of observations from ERT in format nodename
    parameters:    A dict of  parameters from ERT in format nodename:paramname.
                            Key is node name. Values are lists of parameter names for the node.
    grid_config:   A dict with configuration parameters for grid related to
                           FIELD keyword in ERT.
    correlations:   A list of CorrelationConfig objects keeping name of
                            one correlation set which defines the input to
                            create a ministep object.
    """

    observations: List[str]
    parameters: Dict[str, Optional[List[str]]]
    grid_config: Optional[Dict[str, Union[float, int, List[Union[float, int]]]]]
    correlations: List[CorrelationConfig]

    @validator("correlations")
    def validate_correlations(cls, correlations, values):

        for corr in correlations:
            #            print(f"corr.obs_group.add: {corr.obs_group.add}")
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
            check_validation_of_obs_group_ref_point(corr.obs_group.ref_point, values)

            parameters = get_and_validate_parameters(
                corr.param_group, values["parameters"]
            )
            corr.param_group.add = parameters
            corr.param_group.remove = []

            # Check optional parameters for fields
            check_validaton_of_field_scaling(corr.field_scale)

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


def check_validation_of_obs_group_ref_point(ref_point, values):
    key = "grid_config"
    if key in values.keys():
        grid_config = values[key]
        if grid_config is not None:
            ert_grid_dim = grid_config["dimensions"]
            ert_grid_origo = grid_config["grid_origo"]
            ert_grid_rotation = grid_config["rotation"]
            ert_grid_size = grid_config["size"]
            if ref_point is not None:
                check_ref_point_with_grid_data(
                    ref_point,
                    ert_grid_origo,
                    ert_grid_rotation,
                    ert_grid_size,
                )


def check_validaton_of_field_scaling(field_scale):
    valid_scaling_methods = [
        "gaussian_decay",
        "exponential_decay",
    ]

    if field_scale is not None:
        if field_scale.method not in valid_scaling_methods:
            raise ValueError(
                "Specified method name for scaling of fields is: "
                f"{field_scale.method}\n"
                f"Valid method names are: {valid_scaling_methods}"
            )
        if (
            not isinstance(field_scale.method_param, list)
            or len(field_scale.method_param) != 3
        ):
            raise ValueError(
                f"The method: {field_scale.method} for field "
                "correlation scaling has wrongly specified parameters: "
                f"{field_scale.method_param}.\n"
                f"Expecting a list of three parameters: "
                f"[range1, range2, anisotropy azimuth angle in degrees]."
            )
        if (
            field_scale.method_param[0] <= 0.0
            or field_scale.method_param[1] <= 0.0
            or field_scale.method_param[2] < 0.0
            or field_scale.method_param[2] > 360.0
        ):
            raise ValueError(
                f"Scale method: {field_scale.method} is specified with range parameters: "
                f"({field_scale.method_param[0]}, {field_scale.method_param[1]}) "
                f"and anisotropy angle: {field_scale.method_param[2]}\n"
                f"Range parameters must be positive and azimuth angle must be in degrees "
                "between 0 and 360 degrees."
            )


def get_and_validate_parameters(param_group, values):
    params_added = expand_wildcards_with_param(param_group.add, values)
    #            print(f"params_added: {params_added}")
    if param_group.remove is not None:
        params_removed = expand_wildcards_with_param(param_group.remove, values)
        #                print(f"params_removed: {params_removed}")
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
        #            print(f"parameters final: {parameters}\n")

    return parameters
