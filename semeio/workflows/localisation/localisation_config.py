from typing import List, Optional, Union, Dict

from pydantic import BaseModel, validator
from semeio.workflows.localisation.local_script_lib import (
    expand_wildcards,
    expand_wildcards_with_param,
    check_for_duplicated_correlation_specifications,
)


class ObsConfig(BaseModel):
    add: Union[str, List[str]]
    remove: Optional[List[str]]

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


class CorrelationConfig(BaseModel):
    name: str
    obs_group: ObsConfig
    param_group: ParamConfig


class LocalisationConfig(BaseModel):
    """
    observations:  A list of observations from ERT in format nodename
    parameters:    A list of  parameters from ERT in format nodename:paramname
    correlations:   A list of CorrelationConfig objects keeping name of
                            one correlation set which defines the input to
                            create a ministep object.
    """

    observations: List[str]
    parameters: Dict[str, Optional[List[str]]]
    correlations: List[CorrelationConfig]

    @validator("correlations")
    def validate_correlations(cls, correlations, values):
        for corr in correlations:
            print(f"corr.obs_group.add: {corr.obs_group.add}")
            if len(corr.obs_group.add) == 0:
                raise ValueError(
                    "Number of specified observations for correlation: "
                    f"{corr.name} is 0"
                )

            obs_added = expand_wildcards(corr.obs_group.add, values["observations"])
            if corr.obs_group.remove is not None:
                obs_removed = expand_wildcards(
                    corr.obs_group.remove, values["observations"]
                )
                observations = [obs for obs in obs_added if obs not in obs_removed]
            else:
                observations = obs_added
            corr.obs_group.add = observations
            corr.obs_group.remove = []

            params_added = expand_wildcards_with_param(
                corr.param_group.add, values["parameters"]
            )
            print(f"params_added: {params_added}")
            if corr.param_group.remove is not None:
                params_removed = expand_wildcards_with_param(
                    corr.param_group.remove, values["parameters"]
                )
                print(f"params_removed: {params_removed}")
                nodes_to_remove = []
                for node_name, param_names in params_added.items():
                    if node_name in params_removed:
                        param_names_to_remove = params_removed[node_name]
                        param_names_in_node_after_remove = [
                            param
                            for param in param_names
                            if param not in param_names_to_remove
                        ]
                        params_added[node_name] = param_names_in_node_after_remove
                        if len(param_names_in_node_after_remove) == 0:
                            nodes_to_remove.append(node_name)
                for node_name in nodes_to_remove:
                    del params_added[node_name]
                parameters = params_added
            else:
                parameters = params_added
            print(f"parameters final: {parameters}\n")
            corr.param_group.add = parameters
            corr.param_group.remove = []

        number_of_duplicates = check_for_duplicated_correlation_specifications(
            correlations
        )
        if number_of_duplicates > 0:
            raise ValueError(
                "Some correlations are specified multiple times. "
                f"Found {number_of_duplicates} duplicated correlations."
            )
        return correlations
