#!/usr/bin/env python
#
import argparse
import logging
import os
import sys

from semeio import valid_file
from semeio.jobs.nexus import nexus

description = """
Argument is the case name (fcs file without extension)
Reference documents:
Nexus_User_Ref.pdf, version 5000.4.11.1, pp. 75, 82.
NexusReleaseNotes.pdf, version 5000.4.11.1, pp. 44.
"""


def create_parser():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "nexusfile", type=valid_file, help="Path to nexusfile",
    )

    parser.add_argument(
        "eclbase", type=str, help="eclbase",
    )

    parser.add_argument(
        "refcase", type=str, help="refcase",
    )

    parser.add_argument(
        "history", type=str, help="history",
    )
    
    parser.add_argument(
        "--log-level",
        "-l",
        required=False,
        default="WARNING",
        type=logging.getLevelName,
    )

    return parser


def main(args):
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    nexus.run(
        nexusfile=parsed_args.nexusfile,
        eclbase=parsed_args.eclbase,
        refcase=parsed_args.refcase,
        history=parsed_args.history,
        log_level=parsed_args.log_level,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
