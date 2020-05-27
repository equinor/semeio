import yaml

from ert_data.measured import MeasuredData
from ert_shared.libres_facade import LibresFacade
from semeio.communication import SemeioScript

from semeio.jobs import misfit_preprocessor
from semeio.jobs.scripts.correlated_observations_scaling import (
    CorrelatedObservationsScalingJob,
)
from semeio.jobs.correlated_observations_scaling.exceptions import EmptyDatasetException


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
    config = misfit_preprocessor.assemble_config(config_record, measured_data,)
    if not config.valid:
        # The config is loaded by misfit_preprocessor.run first. The
        # second time should never fail!
        raise ValueError("Misfit preprocessor config not valid on second load")

    scale_conf = config.snapshot.scaling
    return {
        "threshold": scale_conf.threshold,
        "std_cutoff": scale_conf.std_cutoff,
        "alpha": scale_conf.alpha,
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
    return MeasuredData(facade, obs_keys)
