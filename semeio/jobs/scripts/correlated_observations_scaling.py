import collections

import yaml
from ert_shared.libres_facade import LibresFacade
from res.enkf import ErtScript

from semeio.jobs.correlated_observations_scaling.job import scaling_job


class CorrelatedObservationsScalingJob(ErtScript):
    def run(self, job_config_file):
        facade = LibresFacade(self.ert())
        user_config = load_yaml(job_config_file)
        user_config = _insert_default_group(user_config)
        for job_config in user_config:
            scaling_job(facade, job_config)


def load_yaml(f_name):
    with open(f_name, "r") as fin:
        config = yaml.safe_load(fin)
    return config


def _insert_default_group(value):
    if isinstance(value, collections.Mapping):
        return [value]
    return value
