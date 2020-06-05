# -*- coding: utf-8 -*-
import logging
import configsuite
import numpy as np
from semeio.jobs.correlated_observations_scaling import job_config
from semeio.jobs.correlated_observations_scaling.exceptions import ValidationError
from semeio.jobs.correlated_observations_scaling.obs_utils import (
    create_active_lists,
    find_and_expand_wildcards,
)
from semeio.jobs.correlated_observations_scaling.scaled_matrix import DataMatrix
from semeio.jobs.correlated_observations_scaling.validator import has_keys, is_subset


class ScalingJob(object):
    def __init__(self, obs_keys, obs, obs_with_data, user_config_dict, reporter):
        """Creates a ScalingJob instance with the given obs_keys, obs,
        obs_with_data and user_config_dict."""
        self._obs = obs
        self._obs_with_data = obs_with_data
        self._obs_keys = obs_keys
        self._config = self._setup_configuration(user_config_dict)
        self._validate()
        self._reporter = reporter

    def scale(self, measured_data):
        """
        Collects data, performs scaling and applies scaling, assumes validated input.
        """
        config = self._config.snapshot

        measured_data.remove_failed_realizations()
        measured_data.remove_inactive_observations()
        measured_data.filter_ensemble_mean_obs(config.CALCULATE_KEYS.alpha)
        measured_data.filter_ensemble_std(config.CALCULATE_KEYS.std_cutoff)

        matrix = DataMatrix(measured_data.data)
        matrix.normalize_by_std()

        scale_factor = matrix.get_scaling_factor(config.CALCULATE_KEYS)
        events = config.CALCULATE_KEYS
        data_matrix = matrix.get_data_matrix()
        nr_components, singular_values = matrix.get_nr_primary_components(
            threshold=events.threshold
        )
        self._reporter.publish("svd", list(singular_values))
        nr_observations = data_matrix.shape[1]

        logging.info("Scaling factor calculated from {}".format(events.keys))

        scale_factor = np.sqrt(nr_observations / float(nr_components))
        self._reporter.publish("scale_factor", scale_factor)

        update_data = create_active_lists(self._obs, config.UPDATE_KEYS.keys)
        self._update_scaling(self._obs, scale_factor, update_data)

    def _validate(self):
        """
        Validates the job. If invalid, raises an ValidationError.
        """
        errors = [] if self._config.valid else self._config.errors
        calc_keys = self.get_calc_keys()
        application_keys = [
            entry.key for entry in self._config.snapshot.UPDATE_KEYS.keys
        ]

        errors.extend(is_subset(calc_keys, application_keys))
        errors.extend(
            has_keys(self._obs_keys, calc_keys, "Key: {} has no observations")
        )
        errors.extend(has_keys(self._obs_with_data, calc_keys, "Key: {} has no data"))
        if len(errors) > 0:
            raise ValidationError("Invalid job", errors)

    def get_calc_keys(self):
        return [event.key for event in self._config.snapshot.CALCULATE_KEYS.keys]

    def get_index_lists(self):
        return [
            event.index if len(event.index) > 0 else None
            for event in self._config.snapshot.CALCULATE_KEYS.keys
        ]

    def _setup_configuration(self, config_data):
        """
        Creates a ConfigSuite instance and inserts default values
        """
        schema = job_config.build_schema()
        config_dict = find_and_expand_wildcards(self._obs_keys, config_data)
        config = configsuite.ConfigSuite(config_dict, schema, deduce_required=True,)
        return config

    @staticmethod
    def _update_scaling(obs, scale_factor, events):
        """
        Applies the scaling factor to the user specified index, SUMMARY_OBS needs to be
        treated differently as it only has one data point per node, compared with other
        observation types which have multiple data points per node.
        """
        for event in events:
            obs_vector = obs[event.key]
            for index, obs_node in enumerate(obs_vector):
                if obs_vector.getImplementationType().name == "SUMMARY_OBS":
                    index_list = (
                        event.index
                        if event.index is not None
                        else range(len(obs_vector))
                    )
                    if index in index_list:
                        obs_node.set_std_scaling(scale_factor)
                elif obs_vector.getImplementationType().name != "SUMMARY_OBS":
                    obs_node.updateStdScaling(scale_factor, event.active_list)
        logging.info(
            "Keys: {} scaled with scaling factor: {}".format(
                [event.key for event in events], scale_factor
            )
        )
