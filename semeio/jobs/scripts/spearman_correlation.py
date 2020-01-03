import argparse

from ert_shared.libres_facade import LibresFacade
from res.enkf import ErtScript

from semeio.jobs.spearman_correlation_job.job import spearman_job


class SpearmanCorrelationJob(ErtScript):
    def run(self, *args):
        facade = LibresFacade(self.ert())

        parser = spearman_job_parser()
        args = parser.parse_args(args)

        spearman_job(facade, args.threshold, args.dry_run)


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
