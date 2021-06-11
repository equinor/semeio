from typing import List, Optional, Union, Dict

from pydantic import BaseModel, validator, ValidationError
from semeio.workflows.localisation.local_script_lib import (
    expand_wildcards,
    expand_wildcards_for_param,
)


class ObsConfig(BaseModel):
    add: List[str]
    remove: Optional[List[str]]


class ParamConfig(BaseModel):
    add: List[Union[str, Dict[str, Union[str, List[str]]]]]
    remove: Optional[List[Union[str, Dict[str, Union[str, List[str]]]]]]

    @validator("add")
    def validate_add(cls, add):
        for idx, elem in enumerate(add):
            if isinstance(elem, str):
                add[idx] = {elem: ["*"]}
                continue

            if not len(elem) == 1:
                raise ValidationError(
                    f"Number of entries in parameter model must be 1, was {elem}"
                )

        return add


class CorrelationConfig(BaseModel):
    name: str
    obs_group: ObsConfig
    model_group: ParamConfig


class LocalisationConfig(BaseModel):
    observations: List[str]
    parameters: Dict[str, Union[None, List[str]]]
    correlations: List[CorrelationConfig]

    @validator("correlations")
    def validate_correlations(cls, correlations, values):
        for corr in correlations:
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

            params_added = expand_wildcards_for_param(
                corr.model_group.add, values["parameters"]
            )
            if corr.model_group.remove is not None:
                params_removed = expand_wildcards_for_param(
                    corr.model_group.remove, values["parameters"]
                )
                parameters = [
                    param for param in params_added if param not in params_removed
                ]
            else:
                parameters = params_added
            corr.model_group.add = parameters
            corr.model_group.remove = []
        return correlations

    # @validate("correlations")
    # def validate_correlations(cls, correlations, values):
    #     check_for_duplicated_correlation_specifications
