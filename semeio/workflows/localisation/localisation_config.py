from datetime import datetime
from typing import List, Optional, Union, Dict
from pydantic import BaseModel, validator
from semeio.workflows.localisation.local_script_lib import expand_wildcards_for_obs, expand_wildcards_for_param

class GroupItemConfig(BaseModel):
    add: Union[List[str], str]
    remove: Optional[Union[List[str], str]]

class ObsConfig(GroupItemConfig):
    pass

class ParamConfig(GroupItemConfig):
    pass

class CorrelationConfig(BaseModel):
    name: str
    obs_group: ObsConfig
    model_group: ParamConfig

class LocalisationConfig(BaseModel):
    observations: List[str]
    parameters: Dict[str, Union[None, List[str]]]
    correlations: List[CorrelationConfig]

    @validator("correlations")
    def validate_correlations(cls, v, values):
        for corr in v:
            obs_added = expand_wildcards_for_obs(corr.obs_group.add, values["observations"])
            if corr.obs_group.remove is not None:
                obs_removed = expand_wildcards_for_obs(corr.obs_group.remove, values["observations"])
                observations = [obs for obs in obs_added if obs not in obs_removed]
            else:
                observations = obs_added
            corr.obs_group.add = observations
            corr.obs_group.remove = []

            params_added = expand_wildcards_for_param(corr.model_group.add, values["parameters"])
            if corr.model_group.remove is not None:
                params_removed = expand_wildcards_for_param(corr.model_group.remove, values["parameters"])
                parameters = [param for param in params_added if param not in params_removed]
            else:
                parameters = params_added
            corr.model_group.add = parameters
            corr.model_group.remove = []
        return v
