#!/usr/bin/env python

from semeio.jobs.overburden_timeshift.ots import ots_run
from semeio.jobs.overburden_timeshift.ots_config import generate_rst_doc
import argparse
from semeio import valid_file

short_description = (
    "Overburden timeshift (OTS) generates evolution of reservoir surfaces "
    "based on eclipse models and seismic velocity volume. If the shift "
    "becomes negative it won't be nullified but only divided "
    "by a constant (currently defaults to 5), ie. Time shifts "
    "linked with compaction are smaller than those linked with stretch. "
    "Input yml needs to contains vintages section, where at "
    "least one of the four categories "
    "for the surface computation needs to be set. "
    "By specifying --params, one gets the full list of yml config parameters."
)

description = short_description + generate_rst_doc()
category = "modeling.surface"


def _get_args_parser():
    parser = argparse.ArgumentParser(description=short_description)

    parser.add_argument(
        "-c", "--config", help="ots config file, yaml format required", type=valid_file
    )
    parser.add_argument(
        "-p",
        "--params",
        action="store_true",
        help="shows all ots config job parameters",
        dest="help_params",
        default=False,
    )
    return parser


def main_entry_point(args=None):
    parser = _get_args_parser()
    options = parser.parse_args(args)
    if options.help_params:
        print(generate_rst_doc())
    else:
        ots_run(options.config)


if __name__ == "__main__":
    main_entry_point()
