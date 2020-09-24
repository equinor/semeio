import argparse
import sys

import pandas as pd
from ert_shared.plugins.plugin_manager import hook_implementation
from fmu import ensemble
from res.job_queue import ErtScript


def csv_exporter(runpathfile, time_index, outputfile, column_keys=None):
    """Export CSV data (summary and parameters) from an EnsembleSet

    The EnsembleSet is described by a runpathfile which must exists
    and point to realizations"""
    ensemble_set = ensemble.EnsembleSet(
        name="ERT EnsembleSet for CSV_EXPORT2", runpathfile=runpathfile
    )
    summary = ensemble_set.load_smry(time_index=time_index, column_keys=column_keys)
    parameters = ensemble_set.parameters
    summary_parameters = pd.merge(summary, parameters)
    summary_parameters.to_csv(outputfile, index=False)


class CsvExport2Job(ErtScript):  # pylint: disable=too-few-public-methods
    def run(self, *args):
        main(args)


def main(args):
    parser = csv_export_parser()
    args = parser.parse_args(args)

    csv_exporter(
        runpathfile=args.runpathfile,
        time_index=args.time_index,
        outputfile=args.outputfile,
        column_keys=args.column_keys,
    )

    print("{} csv-export written to {}".format(args.time_index, args.outputfile))


def csv_export_parser():
    """Setup parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("runpathfile", type=str)
    parser.add_argument("outputfile", type=str)
    parser.add_argument("time_index", type=str, default="monthly")
    parser.add_argument("column_keys", nargs="+", default=None)
    return parser


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(CsvExport2Job, "CSV_EXPORT2")
    workflow.parser = csv_export_parser


def cli():
    main(sys.argv[1:])
