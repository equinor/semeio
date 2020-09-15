import yaml
import configsuite

from ert_data.measured import MeasuredData
from ert_shared.libres_facade import LibresFacade
from ert_shared.plugins.plugin_manager import hook_implementation
from semeio.communication import SemeioScript

from semeio.workflows import misfit_preprocessor
from semeio.workflows.correlated_observations_scaling.cos import (
    CorrelatedObservationsScalingJob,
)
from semeio.workflows.correlated_observations_scaling.exceptions import (
    EmptyDatasetException,
)


class MisfitPreprocessorJob(SemeioScript):  # pylint: disable=too-few-public-methods
    def run(self, *args):
        config_record = _fetch_config_record(args)
        measured_record = _load_measured_record(self.ert())
        scaling_configs = misfit_preprocessor.run(
            **{
                "misfit_preprocessor_config": config_record,
                "measured_data": measured_record,
                "reporter": self.reporter,
            }
        )

        # The execution of COS should be moved into
        # misfit_preprocessor.run when COS no longer depend on self.ert
        # to run.
        scaling_params = _fetch_scaling_parameters(config_record, measured_record)
        for scaling_config in scaling_configs:
            scaling_config["CALCULATE_KEYS"].update(scaling_params)

        try:
            CorrelatedObservationsScalingJob(self.ert()).run(scaling_configs)
        except EmptyDatasetException:
            pass


def _fetch_scaling_parameters(config_record, measured_data):
    config = misfit_preprocessor.assemble_config(
        config_record,
        measured_data,
    )
    if not config.valid:
        # The config is loaded by misfit_preprocessor.run first. The
        # second time should never fail!
        raise ValueError("Misfit preprocessor config not valid on second load")

    scale_conf = config.snapshot.scaling
    return {
        "threshold": scale_conf.threshold,
    }


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


def _load_measured_record(enkf_main):
    facade = LibresFacade(enkf_main)
    obs_keys = [
        facade.get_observation_key(nr) for nr, _ in enumerate(facade.get_observations())
    ]
    measured_data = MeasuredData(facade, obs_keys)
    measured_data.remove_failed_realizations()
    measured_data.remove_inactive_observations()
    measured_data.filter_ensemble_mean_obs(facade.get_alpha())
    measured_data.filter_ensemble_std(facade.get_std_cutoff())
    return measured_data


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(MisfitPreprocessorJob, "MISFIT_PREPROCESSOR")
    _schema = misfit_preprocessor.config._SCHEMA
    rst_doc = configsuite.docs.generate(_schema)
    workflow.description = rst_doc
