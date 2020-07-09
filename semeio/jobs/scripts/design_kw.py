#!/usr/bin/env python

import argparse
import logging
import sys

from semeio import valid_file
from semeio.jobs.design_kw import design_kw

description = """
Performs text->value substitutions (similar to GEN_KW) in templatefile, picking
values from parameters.txt.  This is designed for using in conjunction with a
"Design matrix", but this particular script only requires key-values to exist in
parameters.txt.  Normally, DESIGN2PARAMS is run before DESIGN_KW.

Fails hard if resultfile contains unmatched template directives after
substitutions on non-comment lines (assuming comments start with "--" or "#", at
beginning of line only)
"""

category = "utility.templating"


def create_parser():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "templatefile", type=valid_file, help="Path to template-file",
    )

    parser.add_argument(
        "resultfile", type=str, help="Path to result-file",
    )

    parser.add_argument(
        "--log-level", required=False, default="WARNING", type=logging.getLevelName,
    )

    return parser


def main(args):
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    design_kw.run(
        template_file_name=parsed_args.templatefile,
        result_file_name=parsed_args.resultfile,
        log_level=parsed_args.log_level,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
