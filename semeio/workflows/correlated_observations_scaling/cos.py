import collections

import yaml
import configsuite

from ert_data.measured import MeasuredData
from ert_shared.libres_facade import LibresFacade
from ert_shared.plugins.plugin_manager import hook_implementation
from semeio.communication import SemeioScript
from semeio.workflows.correlated_observations_scaling import job_config
from semeio.workflows.correlated_observations_scaling import ScalingJob
from semeio.workflows.correlated_observations_scaling.obs_utils import keys_with_data


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
            obs,
            obs_keys,
            facade.get_ensemble_size(),
            facade.get_current_fs(),
        )
        default_values = _get_default_values(
            facade.get_alpha(), facade.get_std_cutoff()
        )
        for config in user_config:
            job = ScalingJob(
                obs_keys, obs, obs_with_data, config, self.reporter, default_values
            )
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


def _get_default_values(alpha, std_cutoff):
    return {
        "CALCULATE_KEYS": {"std_cutoff": std_cutoff, "alpha": alpha},
        "UPDATE_KEYS": {},
    }


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(
        CorrelatedObservationsScalingJob, "CORRELATED_OBSERVATIONS_SCALING"
    )
    _schema = job_config.build_schema()
    rst_doc = configsuite.docs.generate(_schema)
    workflow.description = rst_doc
