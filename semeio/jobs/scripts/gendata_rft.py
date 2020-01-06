#!/usr/bin/env python

import argparse
import logging
import sys
import os

from semeio.jobs.rft.utility import (
    load_and_parse_well_time_file,
    valid_eclbase,
    existing_directory,
)
from semeio.jobs.rft import gendata_rft
from semeio.jobs.rft.trajectory import Trajectory
from semeio.jobs.rft.zonemap import ZoneMap

logger = logging.getLogger(__name__)


def _build_parser():
    description = """
The gendata_rft application is used to retrieve rft data from an eclipse output
(or equivalent data format), given a file containing relevant well names and
their corresponding dates. It is also possible to specify a zonemap that validates
each zone specified in the trajectory files. See details of how each file should
be formatted for the individual arguments.

The user will be prompted with a warning and the applications stops if there is
no RFT available for the well at the date.

The pressure is set to -1 and the inactive value to 0 if any of the following
applies:
 - There is not RFT cell related to the trajectory point
 - The trajectory point can not be found in the grid
 - The zone mapping is invalid.
"""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-e",
        "--eclbase",
        type=valid_eclbase,
        required=True,
        help="""
Filepath to an eclipse base (or with equivalent data format) that
contains both RFT and EGRID files. Note that the proper extensions are added,
so it is only necessary to specify the basename
""",
    )
    parser.add_argument(
        "-w",
        "--well_and_time_file",
        type=load_and_parse_well_time_file,
        required=True,
        help="""
Filepath to a file containing the well and time that will be loaded from the RFT
file. The accepted file format is: <well_name> <DD> <MM> <YY> <report>.
For each well name given in the file there must also exist a <well_name.txt> file
that contains a list of trajectory points. Each line must be specified according
to: <utmx> <utmy> <TVD> <MD>
""",
    )
    parser.add_argument(
        "-t",
        "--trajectory_path",
        required=True,
        type=existing_directory,
        help="""
Path to the directory that contains the trajectory files referenced from the well
and time file. Each well must have a trajectory file that contains at least one
trajectory point.
""",
    )
    parser.add_argument(
        "-z",
        "--zonemap",
        required=False,
        type=ZoneMap.load_and_parse_zonemap_file,
        help="""
Path to a file with zonemap information. This file must include every k-value that
any RFT value is located within, irregardless if the trajectory point contains a
zone definition. If a k value does not exist the program will stop. The zone
mapping will be validated and if it is invalid, the pressure is set to -1 and the
inactive value to 0
""",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        required=False,
        default="WARNING",
        type=logging.getLevelName,
        help="Sets the log level",
    )

    return parser


def main_entry_point(args=None):
    arg_parser = _build_parser()
    options = arg_parser.parse_args(args)
    logger.setLevel(options.log_level)

    well_names = [w_info[0] for w_info in options.well_and_time_file]

    trajectories = {
        wname: Trajectory.load_from_file(
            filepath=os.path.join(options.trajectory_path, wname + ".txt")
        )
        for wname in well_names
    }

    logger.info("All files loaded\nRetrieving RFT data...")

    gendata_rft.run(
        well_times=options.well_and_time_file,
        trajectories=trajectories,
        ecl_grid=options.eclbase[0],
        ecl_rft=options.eclbase[1],
        zonemap=options.zonemap,
    )

    logger.info("Completed!")


if __name__ == "__main__":
    main_entry_point(sys.argv[1:])
