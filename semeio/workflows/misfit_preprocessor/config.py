import fnmatch
from typing import List, Dict, Any
from semeio.workflows.misfit_preprocessor.exceptions import ValidationError
from semeio.workflows.misfit_preprocessor.workflow_config import MisfitConfig
import pydantic


def _observations_present(observations, context) -> List[Dict[str, Any]]:
    errors = []
    for observation_key in observations:
        if observation_key not in context:
            errors.append(
                {
                    "loc": ("observations",),
                    "msg": "Found no match for observation {}".format(observation_key),
                }
            )
    return errors


def _realize_filters(observation_keys, context):
    all_keys = tuple(context)
    if len(observation_keys) == 0:
        observation_keys = ("*",)

    matches = set()
    for obs_filter in observation_keys:
        new_matches = set(fnmatch.filter(all_keys, obs_filter))
        if len(new_matches) == 0:
            new_matches = set((obs_filter,))
        matches = matches.union(new_matches)

    return tuple(matches)


def assemble_config(config_data, observation_keys):
    observations = config_data.get("observations")
    if not observations:
        config_data["observations"] = observation_keys
    else:
        config_data["observations"] = _realize_filters(observations, observation_keys)
    errors = _observations_present(config_data["observations"], observation_keys)
    try:
        config = MisfitConfig(**config_data)
    except pydantic.ValidationError as err:
        errors.extend(err.errors())
    if errors:
        raise ValidationError("Invalid configuration of misfit preprocessor", errors)
    return config
