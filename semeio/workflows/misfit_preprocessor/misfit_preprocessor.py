import yaml

from ert_data.measured import MeasuredData
from ert_shared.libres_facade import LibresFacade
from ert_shared.plugins.plugin_manager import hook_implementation
from semeio.communication import SemeioScript

from semeio.workflows.misfit_preprocessor.config import assemble_config
from semeio.workflows import misfit_preprocessor
from semeio.workflows.correlated_observations_scaling.cos import (
    CorrelatedObservationsScalingJob,
)
from semeio.workflows.correlated_observations_scaling.exceptions import (
    EmptyDatasetException,
)
from semeio.workflows.misfit_preprocessor.workflow_config import MisfitConfig


class MisfitPreprocessorJob(SemeioScript):
    def run(self, *args):
        facade = LibresFacade(self.ert())
        config_record = _fetch_config_record(args)
        observations = _get_observations(facade)
        config = assemble_config(config_record, observations)

        measured_record = _load_measured_record(facade, config.observations)
        scaling_configs = misfit_preprocessor.run(
            **{
                "config": config,
                "measured_data": measured_record,
                "reporter": self.reporter,
            }
        )

        # The execution of COS should be moved into
        # misfit_preprocessor.run when COS no longer depend on self.ert
        # to run.

        try:
            CorrelatedObservationsScalingJob(self.ert()).run(scaling_configs)
        except EmptyDatasetException:
            pass


def _fetch_config_record(args):
    if len(args) == 0:
        return {}
    elif len(args) == 1:
        with open(args[0]) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(
            (
                "Excepted at most one argument, namely the path to a "
                "configuration file. Received {} arguments: {}"
            ).format(len(args), args)
        )


def _load_measured_record(facade, obs_keys):
    measured_data = MeasuredData(facade, obs_keys)
    measured_data.remove_failed_realizations()
    measured_data.remove_inactive_observations()
    measured_data.filter_ensemble_mean_obs(facade.get_alpha())
    measured_data.filter_ensemble_std(facade.get_std_cutoff())
    return measured_data


def _get_observations(facade):
    return [
        facade.get_observation_key(nr) for nr, _ in enumerate(facade.get_observations())
    ]


def _get_example(config_example):
    example_string = yaml.dump(config_example, default_flow_style=False)
    return "\n      ".join(example_string.split("\n"))


_INTERIM_DESCRIPTION = f"""
\n\nThe MisfitPreprocessor configuration is currently changing, and will be updated
shortly. It is advised to just run the MISFIT_PREPROCESSOR without a config to
get the default behavior, or a config with just a list of observations under
the observations keyword.


.. code-block:: yaml

    {_get_example({"observations": ["FOPR", "W*"]})}

"""


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(MisfitPreprocessorJob, "MISFIT_PREPROCESSOR")
    schema = MisfitConfig().schema(by_alias=False, ref_template="{model}")
    docs = schema.get("description", "")
    docs += _INTERIM_DESCRIPTION
    workflow.description = docs
    workflow.category = "observations.correlation"
