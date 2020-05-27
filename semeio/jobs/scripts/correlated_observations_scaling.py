import collections

import yaml

from ert_data.measured import MeasuredData
from ert_shared.libres_facade import LibresFacade
from semeio.communication import SemeioScript
from semeio.jobs.correlated_observations_scaling.job import ScalingJob
from semeio.jobs.correlated_observations_scaling.obs_utils import keys_with_data


class CorrelatedObservationsScalingJob(
    SemeioScript
):  # pylint: disable=too-few-public-methods
    def run(self, job_config):
        facade = LibresFacade(self.ert())
        user_config = load_yaml(job_config)
        user_config = _insert_default_group(user_config)

        obs = facade.get_observations()
        obs_keys = [facade.get_observation_key(nr) for nr, _ in enumerate(obs)]
        obs_with_data = keys_with_data(
            obs, obs_keys, facade.get_ensemble_size(), facade.get_current_fs(),
        )

        for config in user_config:
            job = ScalingJob(obs_keys, obs, obs_with_data, config, self.reporter)
            measured_data = MeasuredData(
                facade, job.get_calc_keys(), job.get_index_lists()
            )
            job.scale(measured_data)


def load_yaml(job_config):
    # Allow job_config to be both list and dict.
    if isinstance(job_config, dict) or isinstance(job_config, list):
        return job_config

    with open(job_config, "r") as fin:
        return yaml.safe_load(fin)


def _insert_default_group(value):
    if isinstance(value, collections.Mapping):
        return [value]
    return value
