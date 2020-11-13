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
        step_list = list(obs_vector.getStepList())  # List of steps, 1-indexed
        for step in step_list:
            obs_node = obs_vector.getNode(step)
            if obs_vector.getImplementationType().name == "SUMMARY_OBS":
                index_list = event.index if event.index else [x - 1 for x in step_list]
                if step - 1 in index_list:
                    obs_node.set_std_scaling(scale_factor)
            else:
                obs_node.updateStdScaling(scale_factor, event.active_list)
    logging.info(
        "Keys: {} scaled with scaling factor: {}".format(
            [event.key for event in obs_list], scale_factor
        )
    )
