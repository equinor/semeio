import argparse
import logging
import sys

from semeio import valid_file
from semeio.forward_models.design_kw import design_kw

description = """
Performs text -> value substitutions (similar to GEN_KW) in ``template_file``,
picking values from ``parameters.txt``.

This is designed for using in conjunction with a design matrix that have been
processed using ``DESIGN2PARAMS`` into ``parameters.txt``.

Fails hard if not templates can be replaced by a value.

The prefix (before the colon) in parameter names, as when written by ``GEN_KW``
and not ``DESIGN2PARAMS``, is optional in the template files.

Example: If ``parameters.txt`` has the line ``MULTFLT:FLT_A10_B18 0.001``, both
of the templates ``<MULTFLT:FLT_A10_B18>`` and ``<FLT_A10_B18>`` will expand to
``0.001``.
"""


def create_parser():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "templatefile",
        type=valid_file,
        help="Path to template-file",
    )

    parser.add_argument(
        "resultfile",
        type=str,
        help="Path to result-file",
    )

    parser.add_argument(
        "--log-level",
        required=False,
        default="WARNING",
        type=logging.getLevelName,
    )

    return parser


def main_entry_point() -> None:
    parser = create_parser()
    parsed_args = parser.parse_args()

    valid = design_kw.run(
        template_file_name=parsed_args.templatefile,
        result_file_name=parsed_args.resultfile,
        log_level=parsed_args.log_level,
    )

    if not valid:
        sys.exit(1)
