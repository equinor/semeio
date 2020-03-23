import collections
import argparse
import yaml

from ert_data.measured import MeasuredData
from ert_shared.libres_facade import LibresFacade
from res.enkf import ErtScript
from semeio.jobs.correlated_observations_scaling.exceptions import ValidationError
from semeio.jobs.correlated_observations_scaling.job import ScalingJob
from semeio.jobs.correlated_observations_scaling.obs_utils import keys_with_data


class CorrelatedObservationsScalingJob(ErtScript):
    def run(self, *args):
        facade = LibresFacade(self.ert())

        try:
            parser = scaling_job_parser()
            parsed_args = parser.parse_args(args)
            config = parsed_args.config
        except (KeyError, TypeError):
            config = args[0]

        user_config = load_yaml(config)
        user_config = _insert_default_group(user_config)

        obs = facade.get_observations()
        obs_keys = [facade.get_observation_key(nr) for nr, _ in enumerate(obs)]
        obs_with_data = keys_with_data(
            obs, obs_keys, facade.get_ensemble_size(), facade.get_current_fs(),
        )

        for config in user_config:
            try:
                job = ScalingJob(obs_keys, obs, obs_with_data, config)
            except ValidationError as validation_error:
                print(str(validation_error))
                for error in validation_error.errors:
                    print("\t{}".format(error))
                continue

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


def scaling_job_parser():
    description = """
    correlated_observation_scaling is an ERT workflow job that does Principal
    Component Analysis (PCA) scaling of observations in ERT. The job accepts a
    configuration as the only argument.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "config",
        type=str,
        help="""Obtains a job configuration from this value. A valid job
        configuration is a YAML file in one of three formats: a CALCULATE_KEYS
        and UPDATE_KEYS pair; a list of such pairs; or a singular
        CALCULATE_KEYS directive. Each directive accepts a list of keys that
        each are a key and index hash. Keys can contain wildcards, and indices
        can represent ranges like "1-10,11,12". No index value implies all
        values. CALCULATE_KEYS can be configured using threshold which sets the
        threshold for PCA scaling [default: 0.95]; std_cutoff, the cutoff for
        standard deviation filtering [default: 1e-6]; and alpha for filtering
        between ensemble mean and observations [default: 3].""",
    )
    return parser
