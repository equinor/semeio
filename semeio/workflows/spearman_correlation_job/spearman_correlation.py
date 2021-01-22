import argparse

from ert_data.measured import MeasuredData
from ert_shared.libres_facade import LibresFacade
from ert_shared.plugins.plugin_manager import hook_implementation
from semeio.communication import SemeioScript
from semeio.workflows.correlated_observations_scaling.exceptions import (
    EmptyDatasetException,
)
from semeio.workflows.correlated_observations_scaling.cos import (
    CorrelatedObservationsScalingJob,
)
from semeio.workflows.spearman_correlation_job.job import spearman_job


class SpearmanCorrelationJob(SemeioScript):
    def run(self, *args):
        facade = LibresFacade(self.ert())

        obs_keys = [
            facade.get_observation_key(nr)
            for nr, _ in enumerate(facade.get_observations())
        ]

        measured_data = MeasuredData(facade, obs_keys)

        parser = spearman_job_parser()
        args = parser.parse_args(args)

        scaling_configs = spearman_job(
            measured_data, self.reporter, threshold=args.threshold
        )

        if not args.dry_run:
            try:
                CorrelatedObservationsScalingJob(self.ert()).run(scaling_configs)
            except EmptyDatasetException:
                pass


def spearman_job_parser():
    description = """
    A module that calculates the Spearman correlation in simulated data
    and clusters
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-t",
        "--threshold",
        required=False,
        default=1.15,
        type=float,
        help="""
        Forms flat clusters so that the original
        observations in each flat cluster have no greater a
        cophenetic distance than `t`.
        """,
    )
    parser.add_argument(
        "--output-file",
        required=False,
        type=str,
        help="Name of the outputfile. The format will be yaml.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        required=False,
        help="Dry run, no scaling will be performed",
        action="store_true",
    )
    return parser


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(SpearmanCorrelationJob, "SPEARMAN_CORRELATION")
    workflow.parser = spearman_job_parser
    workflow.category = "observations.correlation"
