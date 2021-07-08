# pylint: disable=E0213
from typing import List, Optional, Union, Dict
from typing_extensions import Literal

from pydantic import BaseModel, validator, confloat
from semeio.workflows.localisation.local_script_lib import (
    expand_wildcards,
    expand_wildcards_with_param,
    check_for_duplicated_correlation_specifications,
    check_ref_point_with_grid_data,
)


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


class ExponentialConfig(BaseModel):
    method: Literal["exponential_decay"]
    main_range: confloat(gt=0)
    perp_range: confloat(gt=0)
    angle: confloat(ge=0, le=360)


class CorrelationConfig(BaseModel):
    name: str
    obs_group: ObsConfig
    param_group: ParamConfig
    ref_point: Optional[List[float]]
    field_scale: Optional[Union[GaussianConfig, ExponentialConfig]]

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
    def validate_workflow(cls, value):
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


class GridInfoConfig(BaseModel):
    grid_dim: List[Union[float, int]]
    origo: List[Union[float, int]]
    grid_rotation: Union[float, int]
    size: List[Union[float, int]]


class LocalisationConfig(BaseModel):
    """
    observations:  A list of observations from ERT in format nodename
    parameters:    A dict of  parameters from ERT in format nodename:paramname.
                            Key is node name. Values are lists of parameter names
                            for the node.
    grid_config:   A dict with configuration parameters for grid related to
                           FIELD keyword in ERT.
    correlations:   A list of CorrelationConfig objects keeping name of
                            one correlation set which defines the input to
                            create a ministep object.
    """

    observations: List[str]
    parameters: Dict[str, Optional[Union[str, List[str]]]]
    node_type: Optional[Dict[str, int]]
    grid_config: Optional[Dict[str, Union[float, int, List[Union[float, int]]]]]
    correlations: List[CorrelationConfig]

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
            #            print(f"values: {values}")
            node_params_dict = values["parameters"]
            node_type_dict = None
            if "node_type" in values.keys():
                node_type_dict = values["node_type"]
            parameters = get_and_validate_parameters(
                corr.param_group, node_params_dict, node_type_dict
            )
            corr.param_group.add = parameters
            corr.param_group.remove = []

            check_validation_of_obs_group_ref_point(values)

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


def check_validation_of_obs_group_ref_point(values):
    key = "grid_config"
    if key in values.keys():
        grid_config = values[key]
        if grid_config is not None:
            #            ert_grid_dim = grid_config["dimensions"]
            ert_grid_origo = grid_config["grid_origo"]
            ert_grid_rotation = grid_config["rotation"]
            ert_grid_size = grid_config["size"]
        key = "ref_point"
        if key in values.keys():
            ref_point = values[key]
            if ref_point is not None:
                check_ref_point_with_grid_data(
                    ref_point,
                    ert_grid_origo,
                    ert_grid_rotation,
                    ert_grid_size,
                )


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
    #        print(f"parameters final: {parameters}\n")

    return parameters
