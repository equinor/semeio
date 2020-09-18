#!/usr/bin/env python3

import argparse
import sys

from res.enkf import ErtScript
from semeio.jobs.csv_export2 import csv_export2
from ert_shared.plugins.plugin_manager import hook_implementation


class CsvExport2Job(ErtScript):  # pylint: disable=too-few-public-methods
    def run(self, *args):
        main(args)


def main(args):
    parser = csv_export_parser()
    args = parser.parse_args(args)

    csv_export2.csv_exporter(
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


if __name__ == "__main__":
    main(sys.argv[1:])


@hook_implementation
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(CsvExport2Job, "CSV_EXPORT2")
    workflow.parser = csv_export_parser
