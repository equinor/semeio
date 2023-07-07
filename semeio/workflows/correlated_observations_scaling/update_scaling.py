import logging

from semeio.workflows.correlated_observations_scaling.obs_utils import (
    create_active_lists,
)


def scale_observations(obs, scale_factor, obs_list):
    """
    Function to scale observations
    :param obs: enkfObservations type
    :param scale_factor: float
    :param obs_list: list of observations to scale, must have fields key and index
    :return: None, scales observations in place
    """
    update_data = create_active_lists(obs, obs_list)
    _update_scaling(obs, scale_factor, update_data)


def _update_scaling(obs, scale_factor, obs_list):
    """
    Applies the scaling factor to the user specified index, SUMMARY_OBS needs to be
    treated differently as it only has one data point per node, compared with other
    observation types which have multiple data points per node.
    """
    for event in obs_list:
        obs_vector = obs[event.key]
        index_list = (
            event.index
            if event.index
            else [x - 1 for x in obs_vector.observations.keys()]
        )
        for step, obs_node in obs_vector.observations.items():
            if obs_vector.observation_type.name == "SUMMARY_OBS":
                if step - 1 in index_list:
                    obs_node.std_scaling = scale_factor
            else:
                obs_node.std_scaling[event.active_list] = scale_factor
    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"Keys: {[event.key for event in obs_list]} scaled "
        f"with scaling factor: {scale_factor}"
    )
